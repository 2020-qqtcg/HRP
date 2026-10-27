#!/usr/bin/env bash
# Local Chromium libraries for this project's Ubuntu 24.04 amd64 host.
# Downloads and extracts packages only; does not install system packages.
set -euo pipefail

native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/vendor/native"
mkdir -p "$native_dir/debs" "$native_dir/root"
cd "$native_dir/debs"

packages=(
  libatk1.0-0t64:amd64=2.52.0-1build1
  libatk-bridge2.0-0t64:amd64=2.52.0-1build1
  libatspi2.0-0t64:amd64=2.52.0-1build1
  libxcomposite1:amd64=1:0.4.5-1build3
  libxdamage1:amd64=1:1.1.6-1build1
  libxfixes3:amd64=1:6.0.0-2build1
  libxrandr2:amd64=2:1.5.2-2build1
  libxrender1:amd64=1:0.9.10-1.1build1
  libxi6:amd64=2:1.8.1-1build1
  libgbm1:amd64=25.2.8-0ubuntu0.24.04.2
  libasound2t64:amd64=1.2.11-1ubuntu0.3
)

# Direct access to the configured Ubuntu mirror works on this host.
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    apt-get -o Acquire::http::Timeout=20 -o Acquire::Retries=0 download "${packages[@]}"

for native_deb in ./*.deb; do
  dpkg-deb -x "$native_deb" "$native_dir/root"
done
printf '%s\n' "${packages[@]}" > "$native_dir/packages.txt"
sha256sum ./*.deb > "$native_dir/SHA256SUMS"
printf 'Local Chromium library directory: %s\n' "$native_dir/root/usr/lib/x86_64-linux-gnu"
