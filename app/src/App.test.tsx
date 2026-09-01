import { cleanup, render, screen, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { App } from "./App";

const fakeApi = {
  listEvents: async () => ({ events: [{ event_id: "demo-industrial-fire", latitude: 20, longitude: 70, prediction: { final_class: "industrial_fire", confidence: .9, model_version: "test" }, evidence: { "FRP change": 2 }, timeline_90d: [{ days_ago: 0, prior_detections: 0 }, { days_ago: 30, prior_detections: 4 }] }] }),
  getEvent: async () => ({ event_id: "demo-industrial-fire", latitude: 20, longitude: 70, prediction: { final_class: "industrial_fire", confidence: .9, model_version: "test" }, evidence: { "FRP change": 2 }, timeline_90d: [{ days_ago: 0, prior_detections: 0 }, { days_ago: 30, prior_detections: 4 }] }),
  getMetrics: async () => ({ stage1: { recall: .91 }, stage2: { macro_f1: .82 }, ranking_promotion: "blocked pending labels" }),
};

describe("THERMIS dashboard", () => {
  afterEach(() => cleanup());

  it("renders a selected event entry", async () => {
    render(<App service={fakeApi} />);
    expect(await screen.findByRole("button", { name: "demo-industrial-fire" })).toBeVisible();
  });

  it("opens evidence, time machine, and evaluation view", async () => {
    const user = userEvent.setup();
    render(<App service={fakeApi} />);
    await user.click(await screen.findByRole("button", { name: /industrial-fire 90%/i }));
    expect(await screen.findByRole("heading", { name: "industrial fire" })).toBeVisible();
    expect(screen.getByText("FRP change")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "7D" }));
    expect(screen.getByRole("region", { name: "thermal history" })).toBeVisible();
    await user.click(within(screen.getByRole("navigation", { name: "dashboard views" })).getByRole("button", { name: "Evaluation" }));
    expect(screen.getByRole("heading", { name: "Promotion readiness" })).toBeVisible();
  });
});
