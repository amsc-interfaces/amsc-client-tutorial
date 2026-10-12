"""RED/GREEN readiness tests for Genesis/direct-IRI alignment (0.7.1)."""
from pathlib import Path
import pytest
REPO = Path(__file__).parent.parent

class TestGenesisAlignment:
    def test_tutorial_url_defaults_genesis_and_rejects_staging(self):
        smoke = (REPO / "scripts" / "smoke_mixed_auth.py").read_text()
        assert "api.genesis.american-science-cloud.org" in smoke
        # Scanner checks exact staging host/phrase, not substring
        assert "staging.api" not in smoke
        assert "STAGING_" not in smoke

    def test_notebooks_via_rig_removed(self):
        assert not (REPO / "notebooks_via_rig").exists()
        readme = (REPO / "README.md").read_text()
        assert "notebooks_via_rig" not in readme

    def test_readme_has_no_rig_references(self):
        readme = (REPO / "README.md").read_text()
        assert "notebooks_via_rig" not in readme

    def test_rig_docs_removed(self):
        current_files = [REPO / "README.md", REPO / "scripts" / "smoke_mixed_auth.py"]
        current_files.extend((REPO / "notebooks").glob("*.ipynb"))
        for path in current_files:
            content = path.read_text().lower()
            assert "notebooks_via_rig" not in content
            assert "facility_via_rig" not in content
            assert "rig.staging.american-science-cloud.org" not in content
            assert "api.staging.american-science-cloud.org" not in content
        assert not (REPO / "notebooks_via_rig").exists()

    def test_openapi_probe_derived_from_args(self):
        smoke = (REPO / "scripts" / "smoke_mixed_auth.py").read_text()
        assert 'openapi_url = f"{args.base_url}/openapi.json"' in smoke
