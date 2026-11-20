$ErrorActionPreference = "Stop"
$ModelName = "finance-masked"
Push-Location $PSScriptRoot
try {
    $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = (Get-Command python -CommandType Application -ErrorAction Stop).Source
    }

    # 1. Train & export Hugging Face model with assistant-only loss masking.
    $TrainArguments = @(
        "train.py",
        "--text-file", ".\examples\shuffled_qa.txt",
        "--tokenizer-format", "gpt2",
        "--output", ".\$ModelName.pt",
        "--export-hf", ".\$ModelName-hf",
        "--device", "cpu",
        "--steps", "600",
        "--eval-interval", "100",
        "--learning-rate", "4e-4",
        "--batch-size", "16",
        "--block-size", "256",
        "--hidden-dim", "192",
        "--layers", "4",
        "--heads", "4"
    )
    if (Test-Path -LiteralPath ".\$ModelName.pt.tokenizer.json") {
        $TrainArguments += @("--tokenizer-file", ".\$ModelName.pt.tokenizer.json")
    }
    & $Python @TrainArguments
    if ($LASTEXITCODE -ne 0) { throw "Training/export failed with exit code $LASTEXITCODE." }

    # 2. Convert the Hugging Face export to F16 GGUF.
    & $Python ..\llama.cpp\convert_hf_to_gguf.py ".\$ModelName-hf" --outfile ".\$ModelName-f16.gguf" --outtype f16
    if ($LASTEXITCODE -ne 0) { throw "GGUF conversion failed with exit code $LASTEXITCODE." }

    # 3. Point the Modelfile at this export, preserving other settings if present.
    $fromLine = "FROM ./$ModelName-f16.gguf"
    if (Test-Path -LiteralPath ".\Modelfile") {
        $modelfileContent = Get-Content ".\Modelfile" -Raw
        if ($modelfileContent -match '(?m)^FROM .+$') {
            $modelfileContent = $modelfileContent -replace '(?m)^FROM .+$', $fromLine
        } else {
            $modelfileContent = "$fromLine`r`n$modelfileContent"
        }
    } else {
        $modelfileContent = @"
$fromLine
TEMPLATE """{{ if .Prompt }}Human: {{ .Prompt }}
Assistant: {{ end }}"""

PARAMETER stop "Human:"
PARAMETER stop "Human"
PARAMETER stop "<|eos|>"
PARAMETER temperature 0.4
PARAMETER top_k 40
PARAMETER num_predict 90
PARAMETER num_ctx 256
"@
    }
    Set-Content -Path ".\Modelfile" -Value $modelfileContent -Encoding Ascii

    # 4. Import into Ollama.
    ollama create $ModelName -f ".\Modelfile"
    if ($LASTEXITCODE -ne 0) { throw "Ollama import failed with exit code $LASTEXITCODE." }

    Write-Host "`nDone! Test your model with: ollama run $ModelName" -ForegroundColor Green
}
finally {
    Pop-Location
}
