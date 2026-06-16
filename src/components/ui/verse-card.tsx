import * as React from "react"

import { cn } from "@/lib/utils"

function VerseCard({
  className,
  reference,
  translation,
  text,
  verseNumber,
  empty = false,
  ...props
}: Omit<React.ComponentProps<"div">, "children"> & {
  reference?: string
  translation?: string
  text?: string
  verseNumber?: number
  empty?: boolean
}) {
  return (
    <div
      data-slot="verse-card"
      className={cn(
        "flex aspect-video w-full flex-col items-center justify-center overflow-hidden rounded-lg bg-gradient-to-br from-[oklch(0.18_0.02_260)] via-[oklch(0.15_0.01_240)] to-[oklch(0.12_0.02_220)]",
        className
      )}
      {...props}
    >
      {empty ? (
        <p className="text-sm text-muted-foreground italic">
          Select a verse to preview
        </p>
      ) : (
        <>
          {reference && (
            <p className="text-center text-xs font-semibold tracking-[0.05em] text-primary uppercase">
              {reference}
              {translation && ` (${translation})`}
            </p>
          )}
          {text && (
            <p className="mt-3 px-8 text-center font-serif text-base leading-relaxed text-white/90">
              {verseNumber != null && (
                <sup className="mr-0.5 align-super text-[0.6rem] text-primary/50">
                  {verseNumber}
                </sup>
              )}
              {text}
            </p>
          )}
        </>
      )}
    </div>
  )
}

export { VerseCard }
