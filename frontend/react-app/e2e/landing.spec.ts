import { expect, test } from "@playwright/test";

test("landing renders without runtime errors or backend calls", async ({
  page,
}) => {
  const errors: string[] = [];
  const apiCalls: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/"))
      apiCalls.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    /Evidence-Based AI.*You Can Verify/,
  );
  await expect(
    page.getByRole("heading", { name: "One platform. Two ways to work." }),
  ).toBeVisible();
  await expect(page.locator("footer")).toBeVisible();
  expect(errors).toEqual([]);
  expect(apiCalls).toEqual([]);
});

test("evidence preview switches to the source and back", async ({ page }) => {
  await page.goto("/");
  const answer = page.getByRole("button", { name: "The answer", exact: true });
  const evidence = page.getByRole("button", { name: "The evidence" });
  await expect(answer).toHaveAttribute("aria-pressed", "true");
  await evidence.click();
  await expect(evidence).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("heading", { name: "The source, right beside the answer." }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: /Return to the grounded answer/ })
    .click();
  await expect(answer).toHaveAttribute("aria-pressed", "true");
});

test("CTA opens the real workspace sign-in screen", async ({
  page,
}) => {
  await page.goto("/");
  const start = page
    .locator(".hero-actions")
    .getByRole("button", { name: "Get started" });
  await start.click();
  await expect(
    page.getByRole("heading", { name: "Welcome back." }),
  ).toBeVisible();
  await expect(page.getByLabel("Email address")).toBeVisible();
  expect(await page.evaluate(() => document.body.style.overflow)).toBe("");
});

test("navigation reaches sections on desktop and mobile", async ({
  page,
}, testInfo) => {
  await page.goto("/");
  if (testInfo.project.name === "mobile") {
    const menu = page.getByRole("button", { name: "Open navigation" });
    await menu.click();
    await expect(
      page.getByRole("button", { name: "Close navigation" }),
    ).toHaveAttribute("aria-expanded", "true");
  }
  await page
    .getByRole("navigation", { name: "Main navigation" })
    .getByRole("link", { name: "Security" })
    .click();
  await expect(page).toHaveURL(/#security$/);
  if (testInfo.project.name === "mobile")
    await expect(
      page.getByRole("button", { name: "Open navigation" }),
    ).toHaveAttribute("aria-expanded", "false");
});

test("narrow layouts do not overflow and reduced motion is respected", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const width of [320, 375, 700, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      `horizontal overflow at ${width}px`,
    ).toBe(true);
  }
  expect(
    await page
      .locator(".hero-copy")
      .evaluate((element) => getComputedStyle(element).animationName),
  ).toBe("none");
});
