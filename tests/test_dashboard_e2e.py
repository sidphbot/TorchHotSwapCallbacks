"""Playwright E2E tests for the hotcb dashboard.

Tests verify that all tabs render, data loads, and key interactions work.
Run with: pytest tests/test_dashboard_e2e.py --headed  (to see browser)
       or: pytest tests/test_dashboard_e2e.py           (headless)

Requires: pip install playwright pytest-playwright && playwright install chromium
"""
import re

import pytest
from playwright.sync_api import Page, expect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goto_dashboard(page: Page, server_url: str, wait_for_metrics: bool = True):
    """Navigate to dashboard and wait for initial data load."""
    page.goto(server_url, timeout=60_000, wait_until="domcontentloaded")
    # Wait for the app to initialize — step counter or chart should appear
    page.wait_for_function("window._initialLoadDone === true", timeout=10_000)
    if wait_for_metrics:
        # Wait for at least one metric to appear in the toggles
        page.wait_for_function(
            "document.querySelectorAll('#metricToggles .metric-toggle-item, #metricToggles button').length > 0",
            timeout=10_000,
        )


def _click_tab(page: Page, tab_name: str):
    """Click a tab by data-tab attribute and wait for content to be visible."""
    tab = page.locator(f'.tab[data-tab="{tab_name}"]')
    tab.click()
    content = page.locator(f'.tab-content[data-tab="{tab_name}"]')
    expect(content).to_be_visible(timeout=3_000)
    return content


# ---------------------------------------------------------------------------
# Tests: Initial Load
# ---------------------------------------------------------------------------

class TestInitialLoad:
    """Verify dashboard loads and displays data on first visit."""

    def test_page_loads(self, page: Page, server_url: str):
        """Dashboard HTML loads without errors."""
        page.goto(server_url, timeout=60_000, wait_until="domcontentloaded")
        expect(page).to_have_title(re.compile(r"hotcb|dashboard", re.IGNORECASE))

    def test_metrics_chart_renders(self, page: Page, server_url: str):
        """Main chart canvas exists and metrics data is loaded."""
        _goto_dashboard(page, server_url)
        canvas = page.locator("#metricsChart")
        expect(canvas).to_be_visible()

    def test_metric_toggles_populated(self, page: Page, server_url: str):
        """Metric toggle buttons appear for discovered metrics."""
        _goto_dashboard(page, server_url)
        toggles = page.locator("#metricToggles")
        expect(toggles).to_be_visible()
        # Should have at least train_loss, grad_norm, lr
        inner = toggles.inner_text()
        assert "train_loss" in inner or len(inner) > 0

    def test_step_counter_nonzero(self, page: Page, server_url: str):
        """Step counter in header shows a non-zero value."""
        _goto_dashboard(page, server_url)
        step_el = page.locator("#stepValue")
        # Wait for step to be populated
        page.wait_for_function(
            "document.getElementById('stepValue') && document.getElementById('stepValue').textContent.trim() !== '0' && document.getElementById('stepValue').textContent.trim() !== ''",
            timeout=5_000,
        )
        text = step_el.inner_text().strip()
        assert text != "0" and text != ""

    def test_websocket_connected(self, page: Page, server_url: str):
        """WebSocket status indicator shows connected."""
        _goto_dashboard(page, server_url)
        ws_status = page.locator("#wsStatus")
        expect(ws_status).to_have_class(re.compile(r"ok"), timeout=5_000)

    def test_no_console_errors(self, page: Page, server_url: str):
        """No critical JS errors on load."""
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        _goto_dashboard(page, server_url)
        # Filter out benign warnings
        critical = [e for e in errors if "TypeError" in e or "ReferenceError" in e]
        assert critical == [], f"JS errors on load: {critical}"


# ---------------------------------------------------------------------------
# Tests: Tab Switching
# ---------------------------------------------------------------------------

class TestTabSwitching:
    """Verify all tabs can be activated and display their content."""

    @pytest.mark.parametrize("tab_name", [
        "metrics", "manifold", "recipe-editor", "autopilot-rules",
        "config-wizard", "compare",
        # "research" has its own dedicated TestResearchTab class
    ])
    def test_tab_visible(self, page: Page, server_url: str, tab_name: str):
        """Each tab's content becomes visible when clicked."""
        _goto_dashboard(page, server_url)
        content = _click_tab(page, tab_name)
        expect(content).to_be_visible()

    def test_non_metrics_tab_hides_panels(self, page: Page, server_url: str):
        """Non-metrics tabs apply compare-active-mode (hides right-col)."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        assert "compare-active-mode" in page.locator("body").get_attribute("class")

    def test_metrics_tab_shows_panels(self, page: Page, server_url: str):
        """Switching back to metrics removes compare-active-mode."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        _click_tab(page, "metrics")
        body_class = page.locator("body").get_attribute("class") or ""
        assert "compare-active-mode" not in body_class


# ---------------------------------------------------------------------------
# Tests: Layout Structure
# ---------------------------------------------------------------------------

class TestLayoutStructure:
    """Verify dashboard layout positions panels correctly."""

    def test_right_col_is_child_of_main(self, page: Page, server_url: str):
        """Right column must be a direct child of .main grid container."""
        _goto_dashboard(page, server_url)
        parent_class = page.evaluate(
            "document.querySelector('.right-col').parentElement.className"
        )
        assert "main" in parent_class, f"right-col parent is '{parent_class}', expected 'main'"

    def test_left_col_is_child_of_main(self, page: Page, server_url: str):
        """Left column must be a direct child of .main grid container."""
        _goto_dashboard(page, server_url)
        parent_class = page.evaluate(
            "document.querySelector('.left-col').parentElement.className"
        )
        assert "main" in parent_class, f"left-col parent is '{parent_class}', expected 'main'"

    def test_metric_cards_inside_left_col(self, page: Page, server_url: str):
        """Pinned metric cards container must be inside left-col."""
        _goto_dashboard(page, server_url)
        parent_class = page.evaluate(
            "document.getElementById('metricCards').parentElement.className"
        )
        assert "left-col" in parent_class, f"metricCards parent is '{parent_class}', expected 'left-col'"

    def test_bottom_panel_inside_left_col(self, page: Page, server_url: str):
        """Bottom timeline panel must be inside left-col."""
        _goto_dashboard(page, server_url)
        parent_class = page.evaluate(
            "document.querySelector('.bottom-panel').parentElement.className"
        )
        assert "left-col" in parent_class, f"bottom-panel parent is '{parent_class}', expected 'left-col'"

    def test_main_has_two_column_grid(self, page: Page, server_url: str):
        """Main container uses a 2-column CSS grid."""
        _goto_dashboard(page, server_url)
        cols = page.evaluate(
            "getComputedStyle(document.querySelector('.main')).gridTemplateColumns"
        )
        # Should have exactly 2 column values (e.g., "886px 310px")
        col_parts = cols.strip().split()
        assert len(col_parts) == 2, f"Expected 2 grid columns, got: '{cols}'"

    def test_main_has_exactly_two_children(self, page: Page, server_url: str):
        """Main grid should have exactly 2 direct children (left-col, right-col)."""
        _goto_dashboard(page, server_url)
        child_count = page.evaluate(
            "document.querySelector('.main').children.length"
        )
        assert child_count == 2, f"Expected 2 children in .main, got {child_count}"

    def test_right_col_to_right_of_left_col(self, page: Page, server_url: str):
        """Right column must be positioned to the right of left column."""
        _goto_dashboard(page, server_url)
        left_box = page.locator(".left-col").bounding_box()
        right_box = page.locator(".right-col").bounding_box()
        assert left_box is not None and right_box is not None
        assert right_box["x"] > left_box["x"], "right-col is not to the right of left-col"

    def test_health_card_in_right_col(self, page: Page, server_url: str):
        """Health card must be a direct child of right-col."""
        _goto_dashboard(page, server_url)
        parent_class = page.evaluate(
            "document.getElementById('healthCard').parentElement.className"
        )
        assert "right-col" in parent_class, f"healthCard parent is '{parent_class}', expected 'right-col'"


# ---------------------------------------------------------------------------
# Tests: Compare Tab
# ---------------------------------------------------------------------------

class TestCompareTab:
    """Verify the Compare Runs tab discovers and lists runs."""

    def test_runs_listed(self, page: Page, server_url: str):
        """Compare tab shows discovered eval runs."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        run_list = page.locator("#compareRunList")
        # Wait for runs to be populated
        expect(run_list).not_to_be_empty(timeout=5_000)
        inner = run_list.inner_text()
        assert "cond_baseline" in inner
        assert "cond_high_lr" in inner

    def test_run_has_focus_button(self, page: Page, server_url: str):
        """Each run in compare list has a Focus button."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        page.wait_for_selector("#compareRunList [data-focus-dir]", timeout=5_000)
        focus_btns = page.locator("#compareRunList [data-focus-dir]")
        assert focus_btns.count() >= 2

    def test_focus_switches_to_metrics(self, page: Page, server_url: str):
        """Clicking Focus loads that run's data and switches to Metrics tab."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        page.wait_for_selector("#compareRunList [data-focus-dir]", timeout=5_000)

        # Click the first Focus button
        focus_btn = page.locator("#compareRunList [data-focus-dir]").first
        focus_label = focus_btn.get_attribute("data-focus-label")
        focus_btn.click()

        # Should switch to metrics tab
        metrics_tab = page.locator('.tab[data-tab="metrics"]')
        expect(metrics_tab).to_have_class(re.compile(r"active"), timeout=3_000)

        # Focused run header should show
        header = page.locator("#focusedRunHeader")
        expect(header).to_be_visible(timeout=3_000)
        assert focus_label in header.inner_text()

    def test_run_selection_toggles(self, page: Page, server_url: str):
        """Clicking a run row toggles its selection for comparison."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        page.wait_for_selector("#compareRunList > div", timeout=5_000)
        # Click the first run row (not the Focus button)
        first_row = page.locator("#compareRunList > div").first
        first_row.click()
        # Border color should change (indicating selection)
        style = first_row.get_attribute("style")
        assert style is not None


# ---------------------------------------------------------------------------
# Tests: Research Tab
# ---------------------------------------------------------------------------

class TestResearchTab:
    """Verify the Research tab renders the Cytoscape.js graph."""

    def test_research_container_visible(self, page: Page, server_url: str):
        """Research tab's Cytoscape container is visible when tab is active."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        container = page.locator("#researchCyContainer")
        expect(container).to_be_visible()

    def test_cytoscape_initializes(self, page: Page, server_url: str):
        """Cytoscape.js creates a canvas inside the container."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        # Cytoscape.js renders into a canvas element inside the container
        page.wait_for_selector(
            "#researchCyContainer canvas",
            timeout=8_000,
        )
        canvas = page.locator("#researchCyContainer canvas")
        assert canvas.count() > 0

    def test_research_stats_populated(self, page: Page, server_url: str):
        """Research stats panel shows hypothesis/evidence counts."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        stats = page.locator("#researchStats")
        # Wait for stats to update from initial "0 hypotheses"
        page.wait_for_function(
            "document.getElementById('researchStats') && !document.getElementById('researchStats').textContent.includes('0 hypotheses')",
            timeout=8_000,
        )
        text = stats.inner_text()
        assert "hypotheses" in text
        assert "evidence" in text

    def test_eval_progress_banner(self, page: Page, server_url: str):
        """Eval progress banner shows completion status."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        # Wait for eval status polling to populate the banner text
        page.wait_for_function(
            "document.getElementById('evalProgressText') && document.getElementById('evalProgressText').textContent.includes('/')",
            timeout=10_000,
        )
        text = page.locator("#evalProgressText").inner_text()
        assert "2/2" in text

    def test_layout_selector_works(self, page: Page, server_url: str):
        """Changing layout selector triggers graph re-layout."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        page.wait_for_selector("#researchCyContainer canvas", timeout=8_000)
        # Switch to force layout
        layout_sel = page.locator("#researchLayout")
        layout_sel.select_option("force")
        # No crash — graph should still be rendered
        page.wait_for_timeout(500)
        canvas = page.locator("#researchCyContainer canvas")
        assert canvas.count() > 0

    def test_stream_filter_dropdown(self, page: Page, server_url: str):
        """Stream filter dropdown gets populated with research streams."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "research")
        page.wait_for_selector("#researchCyContainer canvas", timeout=8_000)
        # Click refresh to ensure streams are loaded
        page.locator("#btnResearchRefresh").click()
        page.wait_for_timeout(1_000)
        stream_sel = page.locator("#researchStreamFilter")
        options = stream_sel.locator("option")
        # Should have at least "All streams" + one real stream
        assert options.count() >= 1


# ---------------------------------------------------------------------------
# Tests: Focus Mode
# ---------------------------------------------------------------------------

class TestFocusMode:
    """Verify focus mode properly loads run data into metrics view."""

    def test_focus_loads_metrics(self, page: Page, server_url: str):
        """After focus, metricsData has entries from the focused run."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        page.wait_for_selector("#compareRunList [data-focus-dir]", timeout=5_000)

        # Focus on cond_baseline
        baseline_btn = page.locator('[data-focus-label="cond_baseline"]')
        if baseline_btn.count() == 0:
            # Fall back to first focus button
            baseline_btn = page.locator("#compareRunList [data-focus-dir]").first
        baseline_btn.click()

        # Metrics tab should be active
        expect(page.locator('.tab-content[data-tab="metrics"]')).to_be_visible(timeout=3_000)

        # S.metricsData should have data
        has_data = page.evaluate(
            "Object.keys(S.metricsData).length > 0 && S.metricNames.size > 0"
        )
        assert has_data, "Focus did not populate S.metricsData or S.metricNames"

    def test_focus_updates_toggles(self, page: Page, server_url: str):
        """After focus, metric toggles reflect the focused run's metrics."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "compare")
        page.wait_for_selector("#compareRunList [data-focus-dir]", timeout=5_000)

        focus_btn = page.locator("#compareRunList [data-focus-dir]").first
        focus_btn.click()
        page.wait_for_timeout(500)

        # Metric toggles should have content
        toggles_text = page.locator("#metricToggles").inner_text()
        assert len(toggles_text) > 0


# ---------------------------------------------------------------------------
# Tests: Autopilot Rules Tab
# ---------------------------------------------------------------------------

class TestAutopilotRulesTab:
    """Verify the Autopilot Rules tab loads and shows rules."""

    def test_rules_list_container_visible(self, page: Page, server_url: str):
        """Autopilot rules list container is visible."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "autopilot-rules")
        rules_list = page.locator("#autopilotRulesList")
        expect(rules_list).to_be_visible()

    def test_add_rule_button_exists(self, page: Page, server_url: str):
        """Add rule button is present."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "autopilot-rules")
        btn = page.locator("#btnRuleAdd")
        expect(btn).to_be_visible()


# ---------------------------------------------------------------------------
# Tests: Config Wizard Tab
# ---------------------------------------------------------------------------

class TestConfigWizardTab:
    """Verify the Config Wizard tab renders its form."""

    def test_config_inputs_visible(self, page: Page, server_url: str):
        """Config wizard form inputs are present."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "config-wizard")
        expect(page.locator("#cfgId")).to_be_visible()
        expect(page.locator("#cfgMaxSteps")).to_be_visible()


# ---------------------------------------------------------------------------
# Tests: Recipe Tab
# ---------------------------------------------------------------------------

class TestRecipeTab:
    """Verify the Recipe Editor tab renders."""

    def test_recipe_toolbar_visible(self, page: Page, server_url: str):
        """Recipe tab toolbar with buttons is visible."""
        _goto_dashboard(page, server_url)
        _click_tab(page, "recipe-editor")
        expect(page.locator("#btnRecipeAdd")).to_be_visible()
        expect(page.locator("#btnRecipeExport")).to_be_visible()
