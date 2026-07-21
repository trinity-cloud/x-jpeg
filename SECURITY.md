# Security policy

Please report vulnerabilities privately through GitHub's security advisory
interface for `trinity-cloud/x-jpeg`. Do not open a public issue containing an
unpatched exploit, malicious model, or crafted input.

Version 0.1.x receives security fixes while it is the current minor release.

Model artifacts are published as Safetensors and pinned by SHA-256. Official
wheels build MozJPEG from a fixed upstream commit and carry its license
notices. X-JPEG still processes untrusted image files through third-party
decoders/encoders; keep Pillow and the packaged wheel updated.

