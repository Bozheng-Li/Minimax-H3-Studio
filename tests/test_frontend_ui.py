#!/usr/bin/env python3
"""在真实 Chromium 里验证前端：工作区切换、Ref2VA 面板、导演台渲染。

API 测试覆盖不到 DOM 和 CSS —— 这次修的 [hidden] 特异性冲突就只在浏览器里
才看得出来。这个脚本把每个改动点都在真实布局中跑一遍，顺手截图。
"""
from __future__ import annotations

import sys
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7860"
SHOTS_DIR = Path("/tmp/h3_ui_shots")

failures: list[str] = []
checks = 0


def cleanup_test_projects() -> None:
    """Remove only projects created by this browser regression test."""
    try:
        with urllib.request.urlopen(BASE + "/api/director/projects", timeout=5) as response:
            projects = json.load(response)
        for project in projects:
            if project.get("title") != "UI 测试临时项目":
                continue
            request = urllib.request.Request(
                BASE + "/api/director/projects/" + project["id"], method="DELETE"
            )
            urllib.request.urlopen(request, timeout=5).close()
    except Exception:
        pass


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        failures.append(label)


def run(width: int, height: int, tag: str) -> None:
    print(f"\n=== 视口 {width}x{height}（{tag}）===")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        page.goto(BASE, wait_until="networkidle")
        page.wait_for_timeout(1500)

        # 测试数据只在当前浏览器上下文中创建，并在结尾清理；这样导演台测试不依赖
        # 用户是否保留了上一次项目，也不会因为空的持久化存档把真实 UI 测试跳过。
        created_project_id = None
        response = page.request.post(BASE + "/api/director/projects", data={
                "title": "UI 测试临时项目",
                "synopsis": "浏览器验收用",
                "visual_bible": "甜恋风格，保持人物身份与场景连续。",
                "shot_plan": "镜头一：两人牵手走过花草坡，女孩轻轻回头并把花别到男孩衣襟。\n\n镜头二：男孩拉住她的手腕转身，把花轻轻别回她的发间。",
                "aspect_ratio": "16:9",
                "default_seconds": 5,
                "quality": "lossless",
                "seed": 42,
                "overlap_seconds": 2,
                "generate_sound": False,
                "auto_approve": False,
        })
        check("可创建导演测试项目", response.ok, response.text)
        if response.ok:
            created_project_id = response.json()["id"]
            project = response.json()
            library = page.request.get(BASE + "/api/library/items").json()
            image = next((item for item in library if item.get("kind") == "image"), None)
            if image:
                imported = page.request.post(BASE + "/api/director/projects/" + created_project_id + "/assets/from-library", data={"item_ids": [image["id"]]})
                check("导演项目可从素材库导入图片", imported.ok, imported.text)
                if imported.ok:
                    asset_id = imported.json()["added_assets"][0]["id"]
                    entity = page.request.post(
                        BASE + "/api/director/projects/" + created_project_id + "/entities",
                        data={
                            "kind": "character",
                            "name": "测试角色卡",
                            "description": "用于验证角色卡排版和删除入口。",
                            "locked_traits": "保持测试角色外观一致。",
                            "asset_ids": [asset_id],
                        },
                    )
                    check("导演项目可创建角色卡", entity.ok, entity.text)
                    for shot in project.get("shots", []):
                        body = {key: shot.get(key) for key in ("title", "prompt", "seconds", "seed_offset", "continuity", "character_ids", "location_id", "start_state", "end_state", "camera", "sound")}
                        body["asset_ids"] = [asset_id]
                        body["scene_asset_id"] = asset_id
                        page.request.put(BASE + "/api/director/projects/" + created_project_id + "/shots/" + shot["id"], data=body)
            page.reload(wait_until="networkidle")
            page.wait_for_timeout(900)

        # 五个工作区互斥可见 —— 修复前 ≤880px 时普通面板隐藏不掉，
        # Ref2VA/导演面板会叠在它下面。
        panel_ids = ["regularPanel", "ref2vaPanel", "directorPanel", "libraryPanel", "outputsPanel"]
        for target, visible in [
            ("ref2va", "ref2vaPanel"),
            ("director", "directorPanel"),
            ("library", "libraryPanel"),
        ]:
            panels = (visible, *[p for p in panel_ids if p != visible])
            page.click(f'.product-mode[data-workspace="{target}"]')
            page.wait_for_timeout(700)
            visible = panels[0]
            check(
                f"切到 {target}：{visible} 可见",
                page.locator(f"#{visible}").is_visible(),
            )
            for hidden in panels[1:]:
                shown = page.locator(f"#{hidden}").is_visible()
                check(f"切到 {target}：{hidden} 已隐藏", not shown,
                      "面板仍然可见，说明 [hidden] 被 @media 规则覆盖了")
            page.screenshot(path=str(SHOTS_DIR / f"{tag}-{target}.png"), full_page=False)

        # 下拉框默认值：fillWorkspaceSelects 用 .value 赋值，reset() 会退回第一项
        page.click('.product-mode[data-workspace="ref2va"]')
        page.wait_for_timeout(400)
        check("Ref2VA 默认 5 秒", page.locator("#refSeconds").input_value() == "5",
              page.locator("#refSeconds").input_value())
        check("Ref2VA 默认 16:9", page.locator("#refRatio").input_value() == "16:9",
              page.locator("#refRatio").input_value())

        page.click('.product-mode[data-workspace="director"]')
        page.wait_for_timeout(900)
        check("导演默认 5 秒", page.locator("#directorSeconds").input_value() == "5",
              page.locator("#directorSeconds").input_value())
        check("导演默认 16:9", page.locator("#directorRatio").input_value() == "16:9",
              page.locator("#directorRatio").input_value())
        check("导演四层页面已渲染", page.locator("[data-director-tab]").count() == 4)

        # 导演台应渲染出项目、镜头卡片与新加的删除入口
        project_rows = page.locator("[data-project]").count()
        check("导演项目列表已渲染", project_rows > 0, f"找到 {project_rows} 个项目")
        if project_rows:
            page.wait_for_timeout(1200)
            shots = page.locator(".shot-card").count()
            check("镜头卡片已渲染", shots > 0, f"找到 {shots} 个")
            check("删除项目按钮存在", page.locator("#directorDelete").count() == 1)
            check("删除镜头按钮存在", page.locator("[data-shot-delete]").count() > 0)
            check("连续性 pending 不标红",
                  page.locator(".shot-card.has-error").count() == 0,
                  f"{page.locator('.shot-card.has-error').count()} 张卡片被标成 error")
            page.screenshot(path=str(SHOTS_DIR / f"{tag}-director-full.png"), full_page=True)

        # 任务队列里 ref2va 要显示成「参考生成」而不是「文生视频」。
        # 走 outputs 而不是 regular —— Ref2VA 分区下普通生成是置灰的，
        # 作品与队列必须在不依赖普通生成的前提下可达。
        page.click('.product-mode[data-workspace="outputs"]')
        page.wait_for_timeout(600)
        check("作品与队列在分区置灰时仍可达", page.locator("#jobList").count() == 1)
        check("outputs 工作区独立于普通生成", not page.locator("#outputsPanel").get_attribute("hidden") is not None)
        page.click('[data-tab="jobs"]')
        page.wait_for_timeout(900)
        job_text = page.locator("#jobList").inner_text()
        if "参考生成" in job_text or "文生视频" in job_text or "图生视频" in job_text:
            check("ref2va 任务标签正确", "参考生成" in job_text,
                  "队列里没有出现「参考生成」，ref2va 仍被显示成别的类型")
        page.screenshot(path=str(SHOTS_DIR / f"{tag}-jobs.png"), full_page=False)

        # 素材库：卡片渲染、筛选、搜索
        page.click('.product-mode[data-workspace="library"]')
        page.wait_for_timeout(1200)
        cards = page.locator("#libGrid .lib-card").count()
        check("素材库卡片已渲染", cards > 0, f"找到 {cards} 张")
        page.click('[data-lib-filter="video"]')
        page.wait_for_timeout(500)
        videos = page.locator("#libGrid .lib-card").count()
        check("按视频筛选生效", videos >= 1, f"筛出 {videos} 张")
        page.fill("#libSearch", "不存在的关键词xyz")
        page.wait_for_timeout(500)
        check("搜索无结果时显示空状态", page.locator("#libGrid .empty").count() == 1)
        page.fill("#libSearch", "")
        page.click('[data-lib-filter="all"]')
        page.wait_for_timeout(400)

        # 分区置灰跟随当前实际模型，不能把测试写死成 Ref2VA。
        partition = page.request.get(BASE + "/api/health").json().get("partition", "ref2va").lower()
        regular_nav = page.locator('.product-mode[data-workspace="regular"]')
        check("普通工作区仍可进入查看队列", not regular_nav.is_disabled())
        check("普通生成提交按钮按分区置灰", page.locator("#submit").is_disabled() == (partition != "fl2va"),
              f"当前分区 {partition}，普通生成按钮状态不匹配")
        check("Ref2VA 提交按钮按分区置灰", page.locator("#refSubmit").is_disabled() == (partition != "ref2va"),
              f"当前分区 {partition}，Ref2VA 按钮状态不匹配")
        for usable in ("ref2va", "director", "library", "outputs"):
            check(f"{usable} 模式可用",
                  not page.locator(f'.product-mode[data-workspace="{usable}"]').is_disabled())

        # 素材库导入入口在 Ref2VA 与导演模式里都要存在
        page.click('.product-mode[data-workspace="ref2va"]')
        page.wait_for_timeout(500)
        check("Ref2VA 有三个素材库导入入口",
              page.locator("[data-ref-lib]").count() == 3,
              f"找到 {page.locator('[data-ref-lib]').count()} 个")
        page.click('.product-mode[data-workspace="director"]')
        page.wait_for_timeout(1200)
        page.click('[data-director-tab="catalog"]')
        page.wait_for_timeout(300)
        check("导演模式有素材库导入入口", page.locator("#directorLibPick").count() == 1)
        check("角色/场景卡有删除入口", page.locator("[data-delete-entity]").count() > 0)
        check("项目素材卡有删除入口", page.locator("[data-delete-asset]").count() > 0)
        if page.locator(".director-world-card").count():
            card_title_size = page.locator(".director-world-body h4").first.evaluate(
                "el => parseFloat(getComputedStyle(el).fontSize)"
            )
            card_body_size = page.locator(".director-world-body p").first.evaluate(
                "el => parseFloat(getComputedStyle(el).fontSize)"
            )
            delete_box = page.locator(".director-card-delete").first.bounding_box()
            check("导演卡片标题字号清晰", card_title_size >= 14, str(card_title_size))
            check("导演卡片正文字号清晰", card_body_size >= 12, str(card_body_size))
            check("删除按钮具备安全点击尺寸", bool(delete_box) and delete_box["width"] >= 32,
                  str(delete_box))

        # 选择器能打开并列出素材
        page.click("#directorLibPick")
        page.wait_for_timeout(900)
        picker = page.locator("#pickerGrid .lib-card").count()
        check("素材选择器列出素材", picker > 0, f"列出 {picker} 个")
        page.click("#pickerCancel")
        page.wait_for_timeout(300)

        real_errors = [e for e in console_errors if "favicon" not in e.lower()]
        check("无 JS 运行时报错", not real_errors, "; ".join(real_errors[:3]))
        check("顶部模型切换卡可见", page.locator("#modelSwitcher").is_visible())
        check("顶部显示当前模型", partition.upper() in page.locator("#modelLabel").inner_text())
        if created_project_id:
            current = page.request.get(
                BASE + "/api/director/projects/" + created_project_id
            ).json()
            removable = next(iter(current.get("assets", [])), None)
            if removable:
                removed = page.request.delete(
                    BASE + "/api/director/projects/" + created_project_id
                    + "/assets/" + removable["id"]
                )
                check("导演素材删除接口可用", removed.ok, removed.text)
                if removed.ok:
                    cleaned = removed.json()
                    check(
                        "删除素材会解除卡片与镜头引用",
                        all(removable["id"] not in e.get("asset_ids", [])
                            for e in cleaned.get("entities", []))
                        and all(removable["id"] not in s.get("asset_ids", [])
                                and s.get("scene_asset_id") != removable["id"]
                                for s in cleaned.get("shots", [])),
                    )
            page.request.delete(BASE + "/api/director/projects/" + created_project_id)
        browser.close()


def main() -> None:
    SHOTS_DIR.mkdir(exist_ok=True)
    cleanup_test_projects()
    try:
        run(1600, 1000, "desktop")
        run(820, 1180, "narrow")  # ≤880px：正是 [hidden] 与 @media 打架的区间
    finally:
        cleanup_test_projects()

    print(f"\n共 {checks} 项检查，失败 {len(failures)} 项")
    if failures:
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)
    print(f"全部通过。截图见 {SHOTS_DIR}")


if __name__ == "__main__":
    main()
