import { useState, useEffect } from "react";
import type { ModelArchitecture, PresetManifestEntry } from "../types/model";

const PRESET_BASE = import.meta.env.BASE_URL + "presets/";

export function useManifest() {
  const [manifest, setManifest] = useState<PresetManifestEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(PRESET_BASE + "manifest.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setManifest(data);
        setError(null);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load manifest:", err);
        setError(`Failed to load model list: ${err.message}`);
        setLoading(false);
      });
  }, []);

  return { manifest, loading, error };
}

export function useModelData(presetId: string | null) {
  const [model, setModel] = useState<ModelArchitecture | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!presetId) {
      setModel(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(PRESET_BASE + presetId + ".json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ModelArchitecture) => {
        setModel(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load model:", err);
        setError(`Failed to load model data: ${err.message}`);
        setModel(null);
        setLoading(false);
      });
  }, [presetId]);

  return { model, loading, error };
}
