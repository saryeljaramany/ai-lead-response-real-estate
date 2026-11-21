(function () {
  "use strict";

  const form = document.getElementById("lead-form");
  const statusEl = document.getElementById("form-status");
  if (!form || !statusEl) {
    return;
  }

  const submitBtn = form.querySelector('button[type="submit"]');

  const MSG_MISSING_TOKEN = "Please complete the verification.";
  const MSG_202 = "Thank you! Your request was received. An agent will contact you shortly.";
  const MSG_400 = "Please check the form: some required details are missing or invalid.";
  const MSG_403 = "The verification failed. Please complete it again and resubmit.";
  const MSG_413 = "Your message is too long.";
  const MSG_429 = "Too many requests. Please wait a minute and try again.";
  const MSG_50X = "The service is temporarily unavailable. Please try again later.";
  const MSG_NETWORK = "Could not reach the server. Check your connection and try again.";
  const MSG_OTHER = "Something went wrong. Please try again later.";

  function resetTurnstile() {
    if (window.turnstile && typeof window.turnstile.reset === "function") {
      try {
        window.turnstile.reset();
      } catch (e) {
        // no-op
      }
    }
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData(form);
    const rawToken = formData.get("cf-turnstile-response");
    const token = typeof rawToken === "string" ? rawToken.trim() : "";

    if (!token) {
      statusEl.textContent = MSG_MISSING_TOKEN;
      return;
    }

    const payload = {
      captcha_token: token
    };

    const fields = [
      "name",
      "email",
      "phone",
      "message",
      "property_type",
      "preferred_location",
      "budget",
      "timeline"
    ];

    for (const field of fields) {
      const val = formData.get(field);
      if (typeof val === "string") {
        const trimmed = val.trim();
        if (trimmed) {
          payload[field] = trimmed;
        }
      }
    }

    if (submitBtn) {
      submitBtn.disabled = true;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(function () {
      controller.abort();
    }, 20000);

    try {
      const response = await fetch("/webhook/lead-intake", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      const status = response.status;
      if (status === 202) {
        statusEl.textContent = MSG_202;
        form.reset();
      } else if (status === 400) {
        statusEl.textContent = MSG_400;
      } else if (status === 403) {
        statusEl.textContent = MSG_403;
      } else if (status === 413) {
        statusEl.textContent = MSG_413;
      } else if (status === 429) {
        statusEl.textContent = MSG_429;
      } else if (status === 502 || status === 503 || status === 504) {
        statusEl.textContent = MSG_50X;
      } else {
        statusEl.textContent = MSG_OTHER;
      }
    } catch (err) {
      statusEl.textContent = MSG_NETWORK;
    } finally {
      clearTimeout(timeoutId);
      if (submitBtn) {
        submitBtn.disabled = false;
      }
      resetTurnstile();
    }
  });
})();
