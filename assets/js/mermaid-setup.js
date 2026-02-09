let mermaidTheme = determineComputedTheme();

/* Create mermaid diagram as another node and hide the code block, appending the mermaid node after it
    this is done to enable retrieving the code again when changing theme between light/dark */
document.addEventListener("readystatechange", () => {
  if (document.readyState === "complete") {
    document.querySelectorAll("pre>code.language-mermaid").forEach((elem) => {
      const svgCode = elem.textContent;
      const backup = elem.parentElement;
      backup.classList.add("unloaded");
      /* create mermaid node */
      let mermaid = document.createElement("pre");
      mermaid.classList.add("mermaid");
      const text = document.createTextNode(svgCode);
      mermaid.appendChild(text);
      backup.after(mermaid);
    });

    mermaid.initialize({ theme: mermaidTheme });

    /* Fix Mermaid viewBox clipping: expand viewBox with padding after render */
    window.addEventListener("load", function () {
      document.querySelectorAll("pre.mermaid svg").forEach(function (svg) {
        var vb = svg.getAttribute("viewBox");
        if (!vb) return;
        var parts = vb.split(/[\s,]+/).map(Number);
        if (parts.length !== 4) return;
        var pad = 120;
        parts[0] -= pad;
        parts[1] -= pad;
        parts[2] += pad * 2;
        parts[3] += pad * 2;
        svg.setAttribute("viewBox", parts.join(" "));
        svg.style.maxWidth = "100%";
      });
    });

    /* Zoomable mermaid diagrams */
    if (typeof d3 !== "undefined") {
      window.addEventListener("load", function () {
        var svgs = d3.selectAll(".mermaid svg");
        svgs.each(function () {
          var svg = d3.select(this);
          svg.html("<g>" + svg.html() + "</g>");
          var inner = svg.select("g");
          var zoom = d3.zoom().on("zoom", function (event) {
            inner.attr("transform", event.transform);
          });
          svg.call(zoom);
        });
      });
    }
  }
});
