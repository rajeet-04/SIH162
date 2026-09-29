import { describe, expect, test } from "bun:test";
import { flareDistanceLabel, nearbyObservations } from "./map-data";

describe("map evidence", () => {
  test("missing distance does not claim a faraway flare", () => {
    for (const distance of [null, undefined, NaN, -1, "100"]) {
      expect(flareDistanceLabel(distance)).toBe("unknown");
    }
    expect(flareDistanceLabel(0)).toBe("0m");
    expect(flareDistanceLabel(30000)).toBe("30.0km");
  });
  test("nearby context excludes target, distant and invalid locations", () => {
    const target = { event_id: "target", latitude: 20, longitude: 75 };
    const near = { event_id: "near", latitude: 20.01, longitude: 75.01 };
    const far = { event_id: "far", latitude: 28, longitude: 78 };
    const invalid = { event_id: "invalid", latitude: NaN, longitude: 75 };
    expect(nearbyObservations([target, near, far, invalid], target)).toEqual([near]);
  });
});
