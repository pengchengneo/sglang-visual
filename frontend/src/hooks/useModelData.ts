import { useState, useEffect } from "react";
import type { ModelArchitecture, PresetManifestEntry } from "../types/model";

const PRESET_BASE = import.meta.env.BASE_URL + "presets/";

export function useManifest() {
  const [manifest, setManifest] = useState<PresetManifestEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(PRESET_BASE + "manifest.json")
      .then((r) => r.json())
      .then((data) => {
        setManifest(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load manifest:", err);
        setLoading(false);
      });
  }, []);

  return { manifest, loading };
}

export function useModelData(presetId: string | null) {
  const [model, setModel] = useState<ModelArchitecture | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!presetId) {
      setModel(null);
      return;
    }
    setLoading(true);
    fetch(PRESET_BASE + presetId + ".json")
      .then((r) => r.json())
      .then((data: ModelArchitecture) => {
        setModel(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load model:", err);
        setLoading(false);
      });
  }, [presetId]);

  return { model, loading };
}
