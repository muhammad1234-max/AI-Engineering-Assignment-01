export function renderErrorPage(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>This page didn't load</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font: 15px/1.6 system-ui, -apple-system, sans-serif; background: #131219; color: #f2ebd9; display: grid; place-items: center; min-height: 100vh; margin: 0; padding: 1.5rem; }
      .card { max-width: 28rem; width: 100%; text-align: center; padding: 2rem; background: #191822; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; }
      h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.5rem; color: #f2ebd9; }
      p { color: #8f8b9e; margin: 0 0 1.5rem; }
      .actions { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
      a, button { padding: 0.5rem 1rem; border-radius: 10px; font: inherit; font-size: 0.875rem; font-weight: 500; cursor: pointer; text-decoration: none; border: 1px solid transparent; }
      .primary { background: #d89e3a; color: #15131e; }
      .secondary { background: #191822; color: #f2ebd9; border-color: rgba(255, 255, 255, 0.12); }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>This page didn't load</h1>
      <p>Something went wrong on our end. You can try refreshing or head back home.</p>
      <div class="actions">
        <button class="primary" onclick="location.reload()">Try again</button>
        <a class="secondary" href="/">Return to Studio</a>
      </div>
    </div>
  </body>
</html>`;
}
