#![expect(clippy::needless_pass_by_value, reason = "Tauri command extractors require pass-by-value")]

use std::collections::HashSet;
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use rhema_detection::{DetectionPipeline, MergedDetection, ReadingMode};

use crate::state::AppState;

/// Confidence assigned to the best FTS5 BM25 match (rank 0) in context search.
pub(crate) const FTS5_RANK0_CONFIDENCE: f64 = 0.75;

/// Confidence decrease per FTS5 rank position.
pub(crate) const FTS5_CONFIDENCE_DECAY: f64 = 0.04;

/// FTS5 results below this confidence are not included.
pub(crate) const FTS5_MIN_CONFIDENCE: f64 = 0.50;

/// Serializable detection result for the frontend
#[derive(Clone, Serialize)]
pub struct DetectionResult {
    pub verse_ref: String,
    pub verse_text: String,
    pub book_name: String,
    pub book_number: i32,
    pub chapter: i32,
    pub verse: i32,
    pub confidence: f64,
    pub source: String,
    pub auto_queued: bool,
    pub transcript_snippet: String,
    /// True when detected from a chapter-only reference (verse defaults to 1, may be refined).
    pub is_chapter_only: bool,
}

fn source_to_string(source: &rhema_detection::DetectionSource) -> String {
    match source {
        rhema_detection::DetectionSource::DirectReference => "direct".to_string(),
        rhema_detection::DetectionSource::Semantic { .. } => "semantic".to_string(),
    }
}

/// Resolve a detection to a full verse result using the database.
///
/// Resolution order:
/// 1. By `verse_id` (semantic detections with DB primary key)
/// 2. By `book_number/chapter/verse_start` with active translation (direct + FTS5 detections)
/// 3. Fallback to unresolved VerseRef fields (no DB available)
pub fn to_result(state: &AppState, merged: &MergedDetection) -> DetectionResult {
    let vr = &merged.detection.verse_ref;
    let vid = merged.detection.verse_id;

    let resolved = state.bible_db.as_ref().and_then(|db| {
        // Try verse_id first (vector-based semantic detections)
        if let Some(id) = vid {
            if let Ok(Some(v)) = db.get_verse_by_id(id) {
                return Some(v);
            }
        }
        // Fall back to book/chapter/verse lookup (direct + FTS5 detections)
        if vr.book_number > 0 && vr.chapter > 0 && vr.verse_start > 0 {
            if let Ok(Some(v)) = db.get_verse(state.active_translation_id, vr.book_number, vr.chapter, vr.verse_start) {
                return Some(v);
            }
        }
        None
    });

    let (reference, verse_text, book_name, book_number, chapter, verse) = match resolved {
        Some(v) => {
            let r = format!("{} {}:{}", v.book_name, v.chapter, v.verse);
            (r, v.text, v.book_name, v.book_number, v.chapter, v.verse)
        }
        None => {
            let r = format!("{} {}:{}", vr.book_name, vr.chapter, vr.verse_start);
            (r, String::new(), vr.book_name.clone(), vr.book_number, vr.chapter, vr.verse_start)
        }
    };

    DetectionResult {
        verse_ref: reference,
        verse_text,
        book_name,
        book_number,
        chapter,
        verse,
        confidence: merged.detection.confidence,
        source: source_to_string(&merged.detection.source),
        auto_queued: merged.auto_queued,
        transcript_snippet: merged.detection.transcript_snippet.clone(),
        is_chapter_only: merged.detection.is_chapter_only,
    }
}

/// Run the detection pipeline on a piece of transcript text
#[tauri::command]
pub fn detect_verses(
    state: State<'_, Mutex<AppState>>,
    pipeline_state: State<'_, Mutex<DetectionPipeline>>,
    text: String,
) -> Result<Vec<DetectionResult>, String> {
    let merged = {
        let mut pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
        pipeline.process(&text)
    };
    let app_state = state.lock().map_err(|e| e.to_string())?;
    let results: Vec<DetectionResult> = merged.iter().map(|m| to_result(&app_state, m)).collect();
    Ok(results)
}

/// Check if semantic search is available
#[tauri::command]
pub fn detection_status(
    pipeline_state: State<'_, Mutex<DetectionPipeline>>,
) -> Result<DetectionStatusResult, String> {
    let pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
    Ok(DetectionStatusResult {
        has_direct: true,
        has_semantic: pipeline.has_semantic(),
        paraphrase_enabled: pipeline.use_synonyms(),
    })
}

/// Toggle paraphrase detection (synonym expansion) on/off
#[tauri::command]
pub fn toggle_paraphrase_detection(
    pipeline_state: State<'_, Mutex<DetectionPipeline>>,
    enabled: bool,
) -> Result<bool, String> {
    let mut pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
    pipeline.set_use_synonyms(enabled);
    log::info!("[DET] Paraphrase detection (synonyms) set to: {enabled}");
    Ok(enabled)
}

#[derive(Serialize)]
pub struct DetectionStatusResult {
    pub has_direct: bool,
    pub has_semantic: bool,
    pub paraphrase_enabled: bool,
}

#[derive(Serialize)]
pub struct SemanticSearchResult {
    pub verse_ref: String,
    pub verse_text: String,
    pub book_name: String,
    pub book_number: i32,
    pub chapter: i32,
    pub verse: i32,
    pub similarity: f64,
}

#[tauri::command]
pub fn semantic_search(
    state: State<'_, Mutex<AppState>>,
    pipeline_state: State<'_, Mutex<DetectionPipeline>>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<SemanticSearchResult>, String> {
    let k = limit.unwrap_or(10);

    // Lock pipeline for vector search (may be slow if ONNX runs)
    let vector_results = {
        let mut pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
        if !pipeline.has_semantic() {
            return Err("Semantic search not available — model or embeddings not loaded".into());
        }
        pipeline.semantic_search(&query, k)
    }; // Pipeline lock dropped

    // Lock AppState for DB lookups only (fast)
    let app_state = state.lock().map_err(|e| e.to_string())?;

    let mut results: Vec<SemanticSearchResult> = vector_results
        .into_iter()
        .filter_map(|(verse_id, similarity)| {
            if let Some(ref db) = app_state.bible_db {
                if let Ok(Some(v)) = db.get_verse_by_id(verse_id) {
                    return Some(SemanticSearchResult {
                        verse_ref: format!("{} {}:{}", v.book_name, v.chapter, v.verse),
                        verse_text: v.text,
                        book_name: v.book_name,
                        book_number: v.book_number,
                        chapter: v.chapter,
                        verse: v.verse,
                        similarity,
                    });
                }
            }
            None
        })
        .collect();

    // FTS5 BM25 across all English translations — resolve to active translation
    if let Some(ref db) = app_state.bible_db {
        let fts_results = db.search_verses_bm25(&query, k).unwrap_or_default();
        let seen: HashSet<(i32, i32, i32)> = results
            .iter()
            .map(|r| (r.book_number, r.chapter, r.verse))
            .collect();

        for (rank, fts) in fts_results.iter().enumerate() {
            if !seen.contains(&(fts.book_number, fts.chapter, fts.verse)) {
                #[expect(clippy::cast_precision_loss, reason = "rank is small")]
                let similarity = FTS5_RANK0_CONFIDENCE - (rank as f64 * FTS5_CONFIDENCE_DECAY);
                if similarity < FTS5_MIN_CONFIDENCE {
                    break;
                }
                // Resolve to active translation text
                if let Ok(Some(v)) = db.get_verse(
                    app_state.active_translation_id,
                    fts.book_number,
                    fts.chapter,
                    fts.verse,
                ) {
                    results.push(SemanticSearchResult {
                        verse_ref: format!("{} {}:{}", v.book_name, v.chapter, v.verse),
                        verse_text: v.text,
                        book_name: v.book_name,
                        book_number: v.book_number,
                        chapter: v.chapter,
                        verse: v.verse,
                        similarity,
                    });
                }
            }
        }
    }

    // Ensure highest similarity is always first
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Get reading mode status
#[tauri::command]
pub fn reading_mode_status(
    state: State<'_, Mutex<ReadingMode>>,
) -> Result<ReadingModeStatus, String> {
    let rm = state.lock().map_err(|e| e.to_string())?;
    Ok(ReadingModeStatus {
        active: rm.is_active(),
        current_verse: rm.current_verse(),
    })
}

#[derive(Serialize)]
pub struct ReadingModeStatus {
    pub active: bool,
    pub current_verse: Option<i32>,
}

/// Stop reading mode
#[tauri::command]
pub fn stop_reading_mode(
    state: State<'_, Mutex<ReadingMode>>,
) -> Result<(), String> {
    let mut rm = state.lock().map_err(|e| e.to_string())?;
    rm.deactivate();
    Ok(())
}

/// Helper to resolve paths to the Qwen embedding model and tokenizer
pub fn resolve_qwen_paths(app: &AppHandle) -> (std::path::PathBuf, std::path::PathBuf) {
    use tauri::Manager;
    let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    
    // 1. Dev environment paths (checking root directory)
    let dev_model = base_dir.join("models/qwen3-embedding-0.6b-int8/model_quantized.onnx");
    let dev_tokenizer = base_dir.join("models/qwen3-embedding-0.6b/tokenizer.json");
    
    // 2. Local App Data directory (where users download it on-demand)
    let local_model = app.path().app_local_data_dir()
        .map(|p| p.join("models/qwen3-embedding-0.6b-int8/model_quantized.onnx")).ok();
    let local_tokenizer = app.path().app_local_data_dir()
        .map(|p| p.join("models/qwen3-embedding-0.6b-int8/tokenizer.json")).ok();
    
    // 3. Production resource directory (if bundled)
    let prod_model = app.path().resource_dir()
        .map(|p| p.join("_up_/models/qwen3-embedding-0.6b-int8/model_quantized.onnx")).ok();
    let prod_tokenizer = app.path().resource_dir()
        .map(|p| p.join("_up_/models/qwen3-embedding-0.6b-int8/tokenizer.json")).ok();
    let prod_tokenizer_fallback = app.path().resource_dir()
        .map(|p| p.join("_up_/models/qwen3-embedding-0.6b/tokenizer.json")).ok();

    // Check Model
    let model_path = if dev_model.exists() {
        dev_model
    } else if local_model.as_ref().map_or(false, |p| p.exists()) {
        local_model.unwrap()
    } else if prod_model.as_ref().map_or(false, |p| p.exists()) {
        prod_model.unwrap()
    } else {
        // Fallback/Default path for downloading
        local_model.unwrap_or(dev_model)
    };

    // Check Tokenizer
    let tokenizer_path = if dev_tokenizer.exists() {
        dev_tokenizer
    } else if local_tokenizer.as_ref().map_or(false, |p| p.exists()) {
        local_tokenizer.unwrap()
    } else if prod_tokenizer.as_ref().map_or(false, |p| p.exists()) {
        prod_tokenizer.unwrap()
    } else if prod_tokenizer_fallback.as_ref().map_or(false, |p| p.exists()) {
        prod_tokenizer_fallback.unwrap()
    } else {
        // Fallback/Default path for downloading
        local_tokenizer.unwrap_or(dev_tokenizer)
    };

    (model_path, tokenizer_path)
}

/// Helper to load Qwen embedding model and pre-computed index into the pipeline
pub fn load_qwen_model_into_pipeline(
    app: &AppHandle,
    pipeline: &mut rhema_detection::DetectionPipeline,
) -> Result<(), String> {
    use tauri::Manager;
    let (model_path, tokenizer_path) = resolve_qwen_paths(app);
    
    if !model_path.exists() || !tokenizer_path.exists() {
        return Err("Qwen model or tokenizer files do not exist".to_string());
    }

    let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let embeddings_path = {
        let dev = base_dir.join("embeddings/kjv-qwen3-0.6b.bin");
        let prod = app.path().resource_dir().map(|p| p.join("_up_/embeddings/kjv-qwen3-0.6b.bin")).ok();
        if dev.exists() {
            dev
        } else if prod.as_ref().map_or(false, |p| p.exists()) {
            prod.unwrap()
        } else {
            dev
        }
    };
    let ids_path = {
        let dev = base_dir.join("embeddings/kjv-qwen3-0.6b-ids.bin");
        let prod = app.path().resource_dir().map(|p| p.join("_up_/embeddings/kjv-qwen3-0.6b-ids.bin")).ok();
        if dev.exists() {
            dev
        } else if prod.as_ref().map_or(false, |p| p.exists()) {
            prod.unwrap()
        } else {
            dev
        }
    };

    use rhema_detection::semantic::embedder::TextEmbedder;
    use rhema_detection::semantic::index::VectorIndex;

    let embedder = rhema_detection::OnnxEmbedder::load(&model_path, &tokenizer_path)
        .map_err(|e| format!("Failed to load ONNX embedder: {e}"))?;

    log::info!("ONNX embedding model loaded successfully from {}", model_path.display());

    if embeddings_path.exists() && ids_path.exists() {
        let dim = embedder.dimension();
        let index = rhema_detection::HnswVectorIndex::load(&embeddings_path, &ids_path, dim)
            .map_err(|e| format!("Failed to load HNSW vector index: {e}"))?;
        
        log::info!("Verse embeddings loaded successfully ({} vectors) from {}", index.len(), embeddings_path.display());
        
        pipeline.set_semantic(
            rhema_detection::SemanticDetector::new(
                Box::new(embedder),
                Box::new(index),
            )
        );
        Ok(())
    } else {
        Err(format!(
            "Pre-computed verse embeddings not found (looked in {} and {})",
            embeddings_path.display(),
            ids_path.display()
        ))
    }
}

/// Check if the Qwen model files are downloaded locally
#[tauri::command]
pub async fn check_qwen_model(app: AppHandle) -> Result<bool, String> {
    let (model_path, tokenizer_path) = resolve_qwen_paths(&app);
    Ok(model_path.exists() && tokenizer_path.exists())
}

/// Download Qwen model and tokenizer sequentially with progress updates
#[tauri::command]
pub async fn download_qwen_model(
    app: AppHandle,
    pipeline_state: State<'_, Mutex<rhema_detection::DetectionPipeline>>,
) -> Result<(), String> {
    use tauri::Manager;
    // 1. Check if model already exists and is loadable
    let (model_path, tokenizer_path) = resolve_qwen_paths(&app);
    if model_path.exists() && tokenizer_path.exists() {
        let mut pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
        if !pipeline.has_semantic() {
            load_qwen_model_into_pipeline(&app, &mut pipeline)?;
        }
        return Ok(());
    }

    // 2. Prepare destination paths
    let local_data_dir = app.path().app_local_data_dir()
        .map_err(|e| format!("Failed to get local data dir: {e}"))?;
    let dest_dir = local_data_dir.join("models").join("qwen3-embedding-0.6b-int8");
    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("Failed to create models directory: {e}"))?;

    let final_model_path = dest_dir.join("model_quantized.onnx");
    let final_tokenizer_path = dest_dir.join("tokenizer.json");

    let temp_model_path = final_model_path.with_extension("tmp");
    let temp_tokenizer_path = final_tokenizer_path.with_extension("tmp");

    // 3. URLs
    let hf_endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let model_url = format!("{}/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/onnx/model_quantized.onnx", hf_endpoint);
    let tokenizer_url = format!("{}/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/tokenizer.json", hf_endpoint);

    let client = reqwest::Client::new();

    log::info!("Fetching headers for model and tokenizer...");
    let model_len = client.head(&model_url).send().await
        .ok().and_then(|r| r.content_length()).unwrap_or(597_990_265);
    let tokenizer_len = client.head(&tokenizer_url).send().await
        .ok().and_then(|r| r.content_length()).unwrap_or(11_423_705);
    
    let total_size = model_len + tokenizer_len;
    let mut downloaded_bytes: u64 = 0;

    // Helper to download a single file with progress tracking
    async fn download_file(
        app: &tauri::AppHandle,
        client: &reqwest::Client,
        url: &str,
        temp_path: &std::path::Path,
        total_size: u64,
        downloaded_bytes: &mut u64,
        last_emit: &mut std::time::Instant,
    ) -> Result<(), String> {
        use tokio::io::AsyncWriteExt;
        use futures_util::StreamExt;

        let response = client.get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to start download from {url}: {e}"))?;

        if !response.status().is_success() {
            return Err(format!("Download from {url} failed with status: {}", response.status()));
        }

        let mut file = tokio::fs::File::create(temp_path)
            .await
            .map_err(|e| format!("Failed to create temp file: {e}"))?;

        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| format!("Error during download: {e}"))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Failed to write to file: {e}"))?;

            *downloaded_bytes += chunk.len() as u64;

            if last_emit.elapsed() > std::time::Duration::from_millis(150) || *downloaded_bytes == total_size {
                let percentage = (*downloaded_bytes as f64 / total_size as f64) * 100.0;
                let _ = app.emit("qwen_download_progress", serde_json::json!({
                    "downloaded": *downloaded_bytes,
                    "total": total_size,
                    "percentage": percentage,
                }));
                *last_emit = std::time::Instant::now();
            }
        }

        file.flush().await.map_err(|e| format!("Failed to flush file: {e}"))?;
        Ok(())
    }

    let mut last_emit = std::time::Instant::now();

    // Download model weights
    log::info!("Downloading Qwen model weights...");
    download_file(
        &app,
        &client,
        &model_url,
        &temp_model_path,
        total_size,
        &mut downloaded_bytes,
        &mut last_emit,
    ).await?;

    // Download tokenizer
    log::info!("Downloading Qwen tokenizer...");
    download_file(
        &app,
        &client,
        &tokenizer_url,
        &temp_tokenizer_path,
        total_size,
        &mut downloaded_bytes,
        &mut last_emit,
    ).await?;

    // Rename temp files
    std::fs::rename(&temp_model_path, &final_model_path)
        .map_err(|e| format!("Failed to rename temp model file: {e}"))?;
    std::fs::rename(&temp_tokenizer_path, &final_tokenizer_path)
        .map_err(|e| format!("Failed to rename temp tokenizer file: {e}"))?;

    log::info!("Qwen embedding model and tokenizer downloaded successfully.");

    // Initialize model in pipeline
    let mut pipeline = pipeline_state.lock().map_err(|e| e.to_string())?;
    load_qwen_model_into_pipeline(&app, &mut pipeline)
        .map_err(|e| format!("Failed to initialize model after download: {e}"))?;

    let _ = app.emit("qwen_download_complete", serde_json::json!({}));
    Ok(())
}
