import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { ChatApp } from "@/components/zaidlm/ChatApp";
import { Splash } from "@/components/zaidlm/Brand";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Shakespearean LLM — Causal Language Model" },
      { name: "description", content: "Chat with Shakespearean LLM, an 809k-parameter Pre-LN Transformer trained from scratch on Tiny Shakespeare." },
      { property: "og:title", content: "Shakespearean LLM — Causal Language Model" },
      { property: "og:description", content: "Chat with an 809k-parameter Pre-LN Transformer trained on Shakespeare." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const [splash, setSplash] = useState(true);
  return (
    <>
      <ChatApp />
      {splash && <Splash onDone={() => setSplash(false)} />}
    </>
  );
}
