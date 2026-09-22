// Progressive enhancement: the complete dictionary remains readable without JavaScript.
const termSearch = document.getElementById("term-search");
if (termSearch) {
  const rows = [...document.querySelectorAll(".document table tbody tr")];
  const output = document.getElementById("term-count");
  function filterTerms() {
    const words = termSearch.value
      .toLocaleLowerCase()
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    let count = 0;
    rows.forEach((row) => {
      const text = row.textContent.toLocaleLowerCase();
      row.hidden = !words.every((word) => text.includes(word));
      if (!row.hidden) count++;
    });
    document.querySelectorAll(".document table").forEach((table) => {
      const empty = ![...table.querySelectorAll("tbody tr")].some(
        (row) => !row.hidden,
      );
      table.hidden = empty;
      let heading = table.previousElementSibling;
      while (heading && heading.tagName !== "H2")
        heading = heading.previousElementSibling;
      if (heading) heading.hidden = empty;
    });
    const contents = document.querySelector(".toc");
    if (contents) contents.hidden = words.length > 0;
    output.textContent = count
      ? `${count} connections`
      : "No matching connection. Try a shorter term.";
  }
  termSearch.addEventListener("input", filterTerms);
  filterTerms();
}
