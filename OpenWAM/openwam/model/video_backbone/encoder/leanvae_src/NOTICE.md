# LeanVAE inference source

Source: https://github.com/westlake-repl/LeanVAE
Commit: 62f7c7d560b83f408b5041f799d70317d9b37713 (MIT; LICENSE included).
Only model, backbone, distribution and wavelet patcher modules are included.
Local changes: relative imports; remove unused beartype and pywt imports.
No Lightning, video dataset, LPIPS or training dependencies are imported.
The production adapter uses weights_only=True and strict=True instead of the
upstream convenience checkpoint loader. Inference math is unchanged.
