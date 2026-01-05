const spotlight = document.querySelector(".bg-spotlight");

document.addEventListener("mousemove", (e) => {
  if (!spotlight) return;

  const isLight = document.body.classList.contains("light");

  spotlight.style.background = isLight
    ? `radial-gradient(circle 220px at ${e.clientX}px ${e.clientY}px,
        rgba(0,0,0,0.08),
        transparent 65%)`
    : `radial-gradient(circle 220px at ${e.clientX}px ${e.clientY}px,
        rgba(255,255,255,0.18),
        transparent 65%)`;
});


document.addEventListener("mousemove", (e) => {
  document.body.style.setProperty("--x", e.clientX + "px");
  document.body.style.setProperty("--y", e.clientY + "px");
});

