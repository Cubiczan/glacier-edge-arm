Output assets/demo.mp4
Set FontSize 20
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 32
Set TypingSpeed 45ms

Hide
Type "cd /Users/cubiczan/Projects/glacier-edge-arm"
Enter
Sleep 500ms
Type "clear"
Enter
Sleep 300ms
Show

Type "# GlacierEdge-Arm"
Sleep 1.5s
Type "# Arm-optimized edge BESS fault detection for data center backup power"
Sleep 2s
Type "# Track: Physical AI | Author: cubiczan | MIT License"
Sleep 3s

Type "cargo run -p gea-cli -- demo --scenario thermal_runaway"
Enter
Sleep 14s

Type "cargo run -p gea-cli -- demo --scenario impedance_fault"
Enter
Sleep 14s

Type "cargo run -p gea-cli -- demo --scenario normal"
Enter
Sleep 14s

Type "cargo run -p gea-cli -- evaluate --windows-per-scenario 100"
Enter
Sleep 16s

Type "cargo run --release -p gea-cli -- bench --iterations 10000"
Enter
Sleep 12s

Type "cargo run -p gea-cli -- policy | head -8"
Enter
Sleep 6s

Type "cargo run -p gea-cli -- audit | tail -1"
Enter
Sleep 5s

Type "# 99.8% accuracy | 3.14x INT8 model reduction | CHP audit trail"
Sleep 3s
Type "# https://codeberg.org/cubiczan/glacier-edge-arm"
Sleep 4s
