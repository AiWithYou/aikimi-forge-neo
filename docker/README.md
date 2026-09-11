<h2 align="center">Aikimi Studio Neo (Docker)</h2>

> [!Warning]
> Requires an **NVIDIA** GPU<br>
> Ensure driver is up to date (`560+` required)

<hr>

## Unraid Deployment

<table>
	<tr>
		<th>Container Path</th>
		<th>Purpose</th>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/models</code>
		</td>
		<td>Checkpoint, Text Encoder, VAE, LoRA, ControlNet</td>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/output</code>
		</td>
		<td>Generated Images</td>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/extensions</code>
		</td>
		<td>User-Installed Extensions</td>
	</tr>
	<tr>
		<td>
			<code>/home/forge/sd-webui/config</code>
		</td>
		<td>User Settings</td>
	</tr>
</table>

- The container runs as **UID 99** / **GID 100** (`nobody:users`) to match Unraid's default share permissions

<hr>

## Building Locally

```bash
git clone --branch neo https://github.com/AiWithYou/aikimi-studio-neo.git
cd aikimi-studio-neo/docker
docker build -t aikimi-neo-local .
```

The container binds to `127.0.0.1` by default. Keep this default for local use.

## Authenticated Remote Mode

Remote binding is an explicit opt-in. Set `AIKIMI_CONTAINER_REMOTE=1` and provide both Gradio and API authentication files. Remote mode fails before launch when either authentication source is missing.

```bash
docker run --gpus all \
  -e AIKIMI_CONTAINER_REMOTE=1 \
  -e 'COMMANDLINE_ARGS=--api --gradio-auth-path /run/secrets/gradio-auth.txt --api-auth-path /run/secrets/api-auth.txt' \
  -v /host/secrets/gradio-auth.txt:/run/secrets/gradio-auth.txt:ro \
  -v /host/secrets/api-auth.txt:/run/secrets/api-auth.txt:ro \
  -p 7860:7860 \
  aikimi-neo-local
```

Each authentication file uses one `username:password` entry per line. Do not bake credentials into the image or place them in the repository. Use a TLS reverse proxy and a firewall when clients connect from outside the trusted LAN.

<hr>

## Pre-Built Image

> a non-official pre-built image is maintained on Docker Hub by [@oromis995](https://github.com/oromis995):

```bash
docker pull oromis995/sd-forge-neo:latest
```

<hr>

## Image Details

<table>
	<tr>
		<td>Base</td>
		<td><code>nvidia/cuda:12.6.3-runtime-ubuntu22.04</code></td>
	</tr>
	<tr>
		<td>Python</td>
		<td><code>3.13</code> via <b>uv</b></td>
	</tr>
	<tr>
		<td>PyTorch</td>
		<td>Latest (<code>cu126</code>)</td>
	</tr>
	<tr>
		<td>User</td>
		<td><code>forge</code> (UID 99 / GID 100)</td>
	</tr>
	<tr>
		<td>Port</td>
		<td>7860</td>
	</tr>
</table>

> [!Note]
> On the first run, `prepare_environment()` will install requirements and dependencies. This may take a few minutes
