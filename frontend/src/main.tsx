import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

const faviconUrl = `${import.meta.env.BASE_URL}los-angeles-dodgers-1.svg`;
const link =
  document.querySelector<HTMLLinkElement>("link[rel='icon']") ||
  document.createElement("link");
link.rel = "icon";
link.type = "image/svg+xml";
link.href = faviconUrl;
document.head.appendChild(link);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
