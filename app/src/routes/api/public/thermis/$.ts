import { createFileRoute } from "@tanstack/react-router";

const FALLBACK_BACKEND = "http://127.0.0.1:8000";

async function proxy(request: Request): Promise<Response> {
  const base = (process.env["THERMIS_API_URL"] ?? FALLBACK_BACKEND).replace(/\/+$/, "");
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/api\/public\/thermis/, "");
  const upstream = `${base}${path}${url.search}`;

  const payload =
    request.method === "GET" || request.method === "HEAD" ? null : await request.text();

  const attempt = () =>
    fetch(upstream, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: payload,
      signal: AbortSignal.timeout(20000),
    });

  // the tunnel occasionally stalls on a cold connection — retry a couple of times
  let response: Response | null = null;
  let lastError: unknown = null;
  for (let i = 0; i < (request.method === "GET" ? 2 : 1); i += 1) {
    try {
      response = await attempt();
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (!response) throw lastError instanceof Error ? lastError : new Error("upstream timeout");

  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
  });
}

export const Route = createFileRoute("/api/public/thermis/$")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        try {
          return await proxy(request);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          return Response.json(
            { error: `THERMIS backend unreachable: ${message}` },
            { status: 502 },
          );
        }
      },
      POST: async ({ request }) => {
        try {
          return await proxy(request);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          return Response.json(
            { error: `THERMIS backend unreachable: ${message}` },
            { status: 502 },
          );
        }
      },
    },
  },
});
