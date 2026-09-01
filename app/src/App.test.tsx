import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it } from "vitest";
import { App } from "./App";

const fakeApi = {
  listEvents: async () => ({ events: [{ event_id: "demo-industrial-fire", latitude: 20, longitude: 70, prediction: { final_class: "industrial_fire", confidence: .9, model_version: "test" }, evidence: { "FRP change": 2 }, timeline_90d: [{ prior_detections: 0 }] }] }),
  getEvent: async () => { throw new Error("unused"); },
};

describe("THERMIS dashboard", () => {
  it("renders a selected event entry", async () => {
    render(<App service={fakeApi} />);
    expect(await screen.findByRole("button", { name: "demo-industrial-fire" })).toBeVisible();
  });
});
