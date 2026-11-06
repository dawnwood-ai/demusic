# How demusic decides there is music

## Pipeline

1. **Extract** the audio (44.1 kHz stereo wav).
2. **Separate** into six stems with Demucs `htdemucs_6s`: vocals, drums, bass, guitar, piano,
   other. Backends: local Demucs, Demucs on fal.ai (your `FAL_KEY`), or a folder of stems you
   made with any other separator.
3. **Gate 1 - loudness.** Mix bass + guitar + piano + other and measure the mean volume. Above
   -50 dBFS = an audible bed. `other` has to be included: string quartets and synth pads land
   there (a wedding-scene string bed measured -64 dBFS on bass/guitar/piano alone).
4. **Gate 2 - classifier.** Subtract the vocals stem from the original mix (dialogue pulls the
   classifier towards "Speech") and run MIT's AST AudioSet classifier on 1 s windows with a
   0.5 s hop. A window is musical when the max probability over 15 music classes (Music,
   Musical instrument, Piano, Synthesizer, Pad, strings, Orchestra, Ambient, Background music,
   Soundtrack, Theme ...) reaches `--threshold` (0.3). Two windows in a row are required:
   a single-window spike is usually an effect (a chime, a door).
5. **Spans.** Musical windows are joined into spans, widened by 0.3 s each side and merged
   when less than 1 s apart.
6. **Strip.** Inside the spans the audio becomes vocals + drums (drums carry footsteps, hits
   and impacts), crossfaded over 0.25 s; outside, the original audio is kept sample for sample
   (then AAC-encoded once). The video stream is copied.

If the classifier fails for any reason, nothing is stripped.

## Calibration

From Dawnwood's production AI-drama clips (Seedance renders):

| rule | false positives on 164 clean clips | found on 87 clips a person de-musiced by ear |
|---|---|---|
| loudness only (> -50 dBFS) | 8 | - |
| loudness AND classifier, threshold 0.3 (default) | 2 | 33 |
| loudness AND classifier, threshold 0.2 | 4 | 41 |

The loudness-only false positives were engine hum, gunfire and room tone, which all land in
Demucs' `other` stem. The clips the detector misses mostly carry a very quiet bed that only
becomes obvious after loudness normalisation; for those use `--threshold 0.2` or `--mode force`.

## Features without torch

The classifier input is Kaldi-compatible log-mel fbanks (128 bins, 25 ms / 10 ms, Hanning,
pre-emphasis 0.97, DC removal) normalised like Hugging Face's `ASTFeatureExtractor`,
implemented in numpy. A test checks it against `torchaudio.compliance.kaldi.fbank`. The model
is the int8 ONNX export (~90 MB), run with onnxruntime on CPU.
