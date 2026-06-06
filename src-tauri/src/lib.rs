mod commands;
mod events;
mod memstats;
mod state;

use std::sync::Mutex;

#[expect(clippy::too_many_lines, reason = "app setup is inherently complex")]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Load .env file — try src-tauri/.env first, then project root ../.env
    dotenvy::dotenv().ok();
    dotenvy::from_filename("../.env").ok();
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(tauri_plugin_log::log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(Mutex::new(state::AppState::new()))
        .manage(Mutex::new(rhema_detection::DetectionPipeline::new()))
        .manage(Mutex::new(rhema_broadcast::ndi::NdiRuntime::default()))
        .manage(Mutex::new(rhema_detection::DirectDetector::new()))
        .manage(Mutex::new(rhema_detection::DetectionMerger::new()))
        .manage(Mutex::new(rhema_detection::ReadingMode::new()))
        .manage(Mutex::new(commands::remote::OscRuntime::new()))
        .manage(Mutex::new(commands::remote::HttpRuntime::new()))
        .invoke_handler(tauri::generate_handler![
            commands::bible::list_translations,
            commands::bible::list_books,
            commands::bible::get_chapter,
            commands::bible::get_verse,
            commands::bible::search_verses,
            commands::bible::get_translation_verses_for_search,
            commands::bible::get_cross_references,
            commands::bible::get_active_translation,
            commands::bible::set_active_translation,
            commands::detection::detect_verses,
            commands::detection::detection_status,
            commands::detection::semantic_search,
            commands::detection::toggle_paraphrase_detection,
            commands::detection::reading_mode_status,
            commands::detection::stop_reading_mode,
            commands::audio::get_audio_devices,
            commands::stt::start_transcription,
            commands::stt::stop_transcription,
            commands::broadcast::list_monitors,
            commands::broadcast::ensure_broadcast_window,
            commands::broadcast::open_broadcast_window,
            commands::broadcast::close_broadcast_window,
            commands::broadcast::start_ndi,
            commands::broadcast::stop_ndi,
            commands::broadcast::get_ndi_status,
            commands::broadcast::push_ndi_frame,
            commands::remote::start_osc,
            commands::remote::stop_osc,
            commands::remote::get_osc_status,
            commands::remote::start_http,
            commands::remote::stop_http,
            commands::remote::get_http_status,
            commands::remote::update_remote_status,
        ])
        .setup(|app| {
            use tauri::Manager;

            memstats::spawn();

            // Try resource dir first (production), then dev fallback
            let db_path = app
                .path()
                .resource_dir()
                .map(|p| p.join("_up_/data/rhema.db"))
                .ok()
                .filter(|p| p.exists())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../data/rhema.db")
                });

            if db_path.exists() {
                let bible_db = rhema_bible::BibleDb::open(&db_path)
                    .expect("Failed to open Bible database");

                let managed_state = app.state::<Mutex<state::AppState>>();
                let mut state = managed_state.lock().unwrap();
                state.bible_db = Some(bible_db);
                drop(state);
                log::info!("Bible database loaded from {}", db_path.display());
            } else {
                log::warn!("Bible database not found at {}", db_path.display());
            }

            // Try to load ONNX embedding model and pre-computed verse index
            // Prefer INT8 quantized model (~571MB) over FP32 (~2.4GB)
            let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
            let model_path = {
                let dev_int8 = base_dir.join("models/qwen3-embedding-0.6b-int8/model_quantized.onnx");
                let dev_fp32 = base_dir.join("models/qwen3-embedding-0.6b/model.onnx");
                let prod_int8 = app.path().resource_dir().map(|p| p.join("_up_/models/qwen3-embedding-0.6b-int8/model_quantized.onnx")).ok();
                let prod_fp32 = app.path().resource_dir().map(|p| p.join("_up_/models/qwen3-embedding-0.6b/model.onnx")).ok();

                if dev_int8.exists() {
                    log::info!("Using INT8 quantized ONNX model (dev)");
                    dev_int8
                } else if dev_fp32.exists() {
                    log::info!("Using FP32 ONNX model (dev)");
                    dev_fp32
                } else if prod_int8.as_ref().map_or(false, |p| p.exists()) {
                    log::info!("Using INT8 quantized ONNX model (prod)");
                    prod_int8.unwrap()
                } else if prod_fp32.as_ref().map_or(false, |p| p.exists()) {
                    log::info!("Using FP32 ONNX model (prod)");
                    prod_fp32.unwrap()
                } else {
                    dev_fp32
                }
            };
            let tokenizer_path = {
                let dev = base_dir.join("models/qwen3-embedding-0.6b/tokenizer.json");
                let prod = app.path().resource_dir().map(|p| p.join("_up_/models/qwen3-embedding-0.6b/tokenizer.json")).ok();
                if dev.exists() {
                    dev
                } else if prod.as_ref().map_or(false, |p| p.exists()) {
                    prod.unwrap()
                } else {
                    dev
                }
            };
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

            if model_path.exists() && tokenizer_path.exists() {
                use rhema_detection::semantic::embedder::TextEmbedder;
                use rhema_detection::semantic::index::VectorIndex;
                match rhema_detection::OnnxEmbedder::load(&model_path, &tokenizer_path) {
                    Ok(embedder) => {
                        log::info!("ONNX embedding model loaded");
                        let managed_pipeline = app.state::<Mutex<rhema_detection::DetectionPipeline>>();
                        let mut pipeline = managed_pipeline.lock().unwrap();

                        // If pre-computed embeddings exist, load the vector index
                        if embeddings_path.exists() && ids_path.exists() {
                            let dim = embedder.dimension();
                            match rhema_detection::HnswVectorIndex::load(&embeddings_path, &ids_path, dim) {
                                Ok(index) => {
                                    log::info!("Verse embeddings loaded ({} vectors)", index.len());
                                    pipeline.set_semantic(
                                        rhema_detection::SemanticDetector::new(
                                            Box::new(embedder),
                                            Box::new(index),
                                        ),
                                    );
                                }
                                Err(e) => {
                                    log::warn!("Failed to load verse embeddings: {e}");
                                }
                            }
                        } else {
                            log::info!("No pre-computed verse embeddings found. Run 'bun run export:verses' then the precompute binary.");
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load ONNX model: {e}");
                    }
                }
            } else {
                log::info!("ONNX model not found. Semantic search disabled. Run 'bun run download:model' to download.");
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
