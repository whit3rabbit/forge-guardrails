cask "forge-guardrails-proxy" do
  arch arm: "arm64", intel: "x86_64"

  version "__VERSION__"
  sha256 arm:   "__ARM_SHA256__",
         intel: "__X86_SHA256__"

  url "https://github.com/whit3rabbit/forge-guardrails/releases/download/v#{version}/forge-guardrails-proxy-#{version}-macos-#{arch}.zip"
  name "forge-guardrails-proxy"
  desc "OpenAI-compatible proxy with Forge tool-call guardrails"
  homepage "https://github.com/whit3rabbit/forge-guardrails"

  binary "forge-guardrails-proxy"
end
