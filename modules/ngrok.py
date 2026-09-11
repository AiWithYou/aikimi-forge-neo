import ngrok

from modules.aikimi_security.redaction import redact_url

# Connect to ngrok for ingress
def connect(token, port, options):
    options = dict(options)
    account = None
    if token is None:
        token = 'None'
    else:
        if ':' in token:
            # token = authtoken:username:password
            token, username, password = token.split(':', 2)
            account = f"{username}:{password}"

    # For all options see: https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py
    if not options.get('authtoken_from_env'):
        options['authtoken'] = token
    if account:
        options['basic_auth'] = account
    if not options.get('session_metadata'):
        options['session_metadata'] = 'stable-diffusion-webui'


    try:
        public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
    except Exception as e:
        print(
            "ngrok connection was not established "
            f"({type(e).__name__}). Verify the configured token source."
        )
    else:
        print(f'ngrok connected to localhost:{port}! URL: {redact_url(public_url)}\n'
               'You can use this link after the launch is complete.')
