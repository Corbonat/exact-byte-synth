# Build script replacement for cmake when only g++ is available.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1
#   powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1 -Configuration ReleaseLto
#   powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1 -Configuration ReleaseLto -Pgo generate -OutputDir cpp_port\build_pgo_gen
#   # then run build_pgo_gen\synth_benchmark.exe on a training workload
#   powershell -ExecutionPolicy Bypass -File cpp_port\build.ps1 -Configuration ReleaseLto -Pgo use -OutputDir cpp_port\build_pgo_use

param(
    [string]$Configuration = "Release",
    [string]$OutputDir = "cpp_port\build",
    [ValidateSet("none", "generate", "use")]
    [string]$Pgo = "none",
    # For -Pgo use: directory containing the training .gcda files. Defaults to
    # OutputDir, so if you generated into build_pgo_gen you should pass
    # -ProfileDir cpp_port\build_pgo_gen explicitly when calling -Pgo use.
    [string]$ProfileDir = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\msys64\ucrt64\bin\g++.exe") {
        $env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
    } else {
        throw "g++ not found in PATH and no MSYS2 ucrt64 toolchain detected."
    }
}

$include = Join-Path $scriptDir "include"
$thirdParty = Join-Path $scriptDir "third_party"
$src = Join-Path $scriptDir "src"
$apps = Join-Path $scriptDir "apps"
$tests = Join-Path $scriptDir "tests"
$build = Join-Path $repoRoot $OutputDir

New-Item -ItemType Directory -Force -Path $build | Out-Null

if ($Pgo -eq "use") {
    # Require at least one .gcda file in OutputDir; bare -fprofile-use expects
    # profile data next to the exe output path.
    $profileFiles = Get-ChildItem $build -Filter *.gcda -ErrorAction SilentlyContinue
    if (-not $profileFiles) {
        throw "No .gcda profile files in $build. Run with -Pgo generate and execute the training workload first (keep the same -OutputDir)."
    }
}

$flagsCommon = @(
    "-std=c++20",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-static-libstdc++",
    "-static-libgcc",
    "-I`"$include`"",
    "-I`"$thirdParty`""
)

switch ($Configuration) {
    "Release" {
        $flagsCommon += @("-O2", "-DNDEBUG", "-mavx2", "-mbmi", "-mbmi2")
    }
    "ReleaseLto" {
        $flagsCommon += @("-O2", "-DNDEBUG", "-mavx2", "-mbmi", "-mbmi2", "-flto=auto", "-fno-fat-lto-objects")
    }
    "ReleaseFast" {
        $flagsCommon += @("-O3", "-DNDEBUG", "-march=native", "-flto=auto")
    }
    "Debug" {
        $flagsCommon += @("-O0", "-g")
    }
    default {
        throw "Unknown Configuration '$Configuration'."
    }
}

switch ($Pgo) {
    "generate" {
        # Bare -fprofile-generate: .gcda files are written next to each
        # compiled target (i.e., inside $build). Avoids path-escaping quirks
        # that GCC applies to absolute Windows paths in -fprofile-generate=<path>.
        $flagsCommon += @("-fprofile-generate")
    }
    "use" {
        # With bare -fprofile-use GCC looks for .gcda files next to where each
        # object/exe would be produced by *this* invocation, which matches the
        # exact paths the -Pgo generate build used. -fprofile-correction
        # tolerates minor mismatches (e.g., inlining differences under LTO).
        #
        # Note: we intentionally do *not* pass -fprofile-use=<dir>: on MinGW
        # that concatenates the dir with the absolute object path and fails
        # to locate the data. Use the same OutputDir for generate and use.
        $flagsCommon += @("-fprofile-use", "-fprofile-correction")
    }
}

$coreSources = @(
    (Join-Path $src "core.cpp"),
    (Join-Path $src "benchmarks.cpp"),
    (Join-Path $src "current_oracle.cpp")
)

$targets = @(
    @{ Name = "synth_search";     Source = (Join-Path $apps "synth_search.cpp") },
    @{ Name = "synth_codegen";    Source = (Join-Path $apps "synth_codegen.cpp") },
    @{ Name = "synth_benchmark";  Source = (Join-Path $apps "synth_benchmark.cpp") },
    @{ Name = "synth_tests";      Source = (Join-Path $tests "synth_tests.cpp") }
)

$start = Get-Date
foreach ($target in $targets) {
    $exe = Join-Path $build ($target.Name + ".exe")
    $args = @() + $flagsCommon + $coreSources + @($target.Source) + @("-o", $exe)
    Write-Host ("building {0}" -f $target.Name)
    & g++ @args
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for $($target.Name) (exit=$LASTEXITCODE)"
    }
}
$elapsed = (Get-Date) - $start
$label = $Configuration
if ($Pgo -ne "none") { $label = "$Configuration+PGO:$Pgo" }
Write-Host ("all targets built in {0:N1}s ({1})" -f $elapsed.TotalSeconds, $label)
