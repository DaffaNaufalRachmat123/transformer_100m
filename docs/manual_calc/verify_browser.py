"""Uji interaksi halaman offline dengan Playwright Chromium.

    python3 docs/manual_calc/verify_browser.py

Memerlukan playwright dan browser Chromium dari Playwright.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto((HERE / "manual_calc.html").as_uri())
    page.wait_for_function("window.ManualDemo && ManualDemo.players.length === 32")
    assert not page.evaluate('ManualDemo.embeddingReady')
    assert page.locator('#generate-weights').is_disabled()
    assert page.locator('[data-animation="5.5"] .story-next').is_disabled()
    assert not page.locator('[data-animation="4"] .story-grid').is_visible()
    page.evaluate("""() => {
        for (const p of ManualDemo.players.slice(0,4)) {
            if (!ManualDemo.isUnlocked(p)) throw Error('Proses sebelumnya belum membuka tahap ini');
            while (p.index < p.frames.length-1) p.nextBtn.click();
        }
    }""")
    assert page.locator('#generate-weights').is_enabled()
    assert page.locator('[data-animation="5.1"] .story-play').is_disabled()
    old = page.evaluate("ManualDemo.model.P.E[0][0]")
    page.locator('#weight-seed').fill('43')
    page.locator('#generate-weights').click()
    page.wait_for_function("ManualDemo.revision === 1")
    assert page.evaluate('ManualDemo.embeddingReady')
    assert page.locator('[data-animation="5.1"] .story-next').is_disabled()
    assert 'belum dibuat' in page.locator('[data-animation="4"] .story-stage').inner_text()
    count = page.evaluate("""() => {
        let n=0;
        for (const p of ManualDemo.players.slice(4)) {
            if (!ManualDemo.isUnlocked(p)) throw Error('Gagal membuka '+p.key);
            while(p.index < p.frames.length-1) {p.nextBtn.click();n++;}
        }
        return n;
    }""")
    assert count > 6000
    print(f"{count} operasi Next berhasil berurutan; embedding wajib di-generate pengguna.")

    for i in range(14):
        page.evaluate("""i => {
            document.documentElement.style.scrollBehavior = 'auto';
            window.scrollTo(0, document.getElementById('s' + i).offsetTop - 40);
        }""", i)
        page.wait_for_timeout(80)
        assert page.locator('nav a[aria-current="location"]').get_attribute("href") == f"#s{i}"
    print("Sidebar mengikuti 14 section.")

    player = '[data-animation="5.5"]'
    page.locator(player + ' .story-reset').click()
    page.locator(player + ' .story-pace input').fill('30')
    page.locator(player + ' .story-play').click()
    page.wait_for_function("ManualDemo.players.find(p => p.key === '5.5').index >= 3")
    page.locator(player + ' .story-play').click()
    paused = page.evaluate("ManualDemo.players.find(p => p.key === '5.5').index")
    page.wait_for_timeout(150)
    assert page.evaluate("ManualDemo.players.find(p => p.key === '5.5').index") == paused
    page.locator(player + ' .story-seek').evaluate("el => {el.value = 100; el.dispatchEvent(new Event('input'));}")
    assert page.evaluate("ManualDemo.players.find(p => p.key === '5.5').index") == 100
    page.locator(player + ' .story-reset').click()
    assert page.evaluate("ManualDemo.players.find(p => p.key === '5.5').index") == 0
    print("Putar, jeda, tempo, seek, dan ulangi berfungsi.")

    new = page.evaluate("ManualDemo.model.P.E[0][0]")
    assert old != new
    assert page.locator('.snapshot').count() == 8
    assert page.locator('.stats .stat b').nth(3).inner_text() == '2/2'
    # Seed yang sama harus memberikan hasil identik.
    page.locator('#generate-weights').click()
    page.wait_for_function("ManualDemo.revision === 2")
    assert page.evaluate("ManualDemo.model.P.E[0][0]") == new
    assert page.locator('[data-animation="5.5"] .story-next').is_disabled()
    assert not page.evaluate("ManualDemo.completed.has('4')")
    # Slider tidak boleh melewati operasi yang belum pernah dilihat.
    page.locator('[data-animation="4"] .story-seek').evaluate("el => {el.value=100;el.dispatchEvent(new Event('input'));}")
    assert page.evaluate("ManualDemo.players.find(p=>p.key==='4').index") == 0
    page.locator('#restore-weights').click()
    page.wait_for_function("ManualDemo.revision === 3")
    assert page.evaluate("ManualDemo.model.P.E[0][0]") == old
    print("Generate, training, inferensi, seed deterministik, dan restore berhasil.")

    page.set_viewport_size({"width": 390, "height": 844})
    for i in (0, 4, 5, 7, 11):
        page.evaluate("i => window.scrollTo(0, document.getElementById('s' + i).offsetTop - 40)", i)
        page.wait_for_timeout(80)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert not errors, errors
    print("Mobile tidak overflow; tidak ada error JavaScript.")
    browser.close()
