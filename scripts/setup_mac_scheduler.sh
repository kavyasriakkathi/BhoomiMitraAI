#!/bin/bash
# BhoomiMitra AI — Setup automated mandi sync on macOS using launchd
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PLIST_PATH="$HOME/Library/LaunchAgents/com.bhoomimitra.mandisync.plist"
PYTHON_BIN="$(which python3)"

mkdir -p "$DIR/logs"

cat << EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.bhoomimitra.mandisync</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$DIR/scripts/sync_mandi_prices.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$DIR</string>
    <key>StandardOutPath</key>
    <string>$DIR/logs/mandi_sync.log</string>
    <key>StandardErrorPath</key>
    <string>$DIR/logs/mandi_sync_error.log</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict>
            <key>Hour</key>
            <integer>7</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>13</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
    </array>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ Mandi sync scheduler registered successfully!"
echo "⏰ Runs automatically every day at 07:00 AM and 01:00 PM IST."
echo "📄 Logs will be written to $DIR/logs/mandi_sync.log"
