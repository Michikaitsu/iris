// Search functionality for IRIS documentation

let searchIndex = []
let searchTimeout

// Load search index
async function loadSearchIndex() {
  try {
    const response = await fetch("/docs/assets/data/search-index.json")
    const data = await response.json()
    searchIndex = data.pages
  } catch (error) {
    console.error("[v0] Error loading search index:", error)
  }
}

// Fuzzy search function
function fuzzySearch(query, text) {
  query = query.toLowerCase()
  text = text.toLowerCase()

  // Exact match gets highest score
  if (text.includes(query)) {
    return text.indexOf(query) === 0 ? 100 : 80
  }

  // Fuzzy matching
  let score = 0
  let queryIndex = 0

  for (let i = 0; i < text.length && queryIndex < query.length; i++) {
    if (text[i] === query[queryIndex]) {
      score += 1
      queryIndex++
    }
  }

  return queryIndex === query.length ? score : 0
}

// Search function
function performSearch(query) {
  if (!query || query.trim().length < 2) {
    showEmptyState()
    return
  }

  const results = []
  const queryLower = query.toLowerCase().trim()

  searchIndex.forEach((page) => {
    let score = 0
    const matchedIn = []

    // Search in title (highest priority)
    const titleScore = fuzzySearch(queryLower, page.title)
    if (titleScore > 0) {
      score += titleScore * 3
      matchedIn.push("title")
    }

    // Search in description
    const descScore = fuzzySearch(queryLower, page.description)
    if (descScore > 0) {
      score += descScore * 2
      matchedIn.push("description")
    }

    // Search in keywords
    page.keywords.forEach((keyword) => {
      const keywordScore = fuzzySearch(queryLower, keyword)
      if (keywordScore > 0) {
        score += keywordScore * 2.5
        matchedIn.push("keyword")
      }
    })

    // Search in content
    const contentScore = fuzzySearch(queryLower, page.content)
    if (contentScore > 0) {
      score += contentScore
      matchedIn.push("content")
    }

    if (score > 0) {
      results.push({
        ...page,
        score,
        matchedIn: [...new Set(matchedIn)],
      })
    }
  })

  // Sort by score
  results.sort((a, b) => b.score - a.score)

  displayResults(results, query)
}

// Display search results
function displayResults(results, query) {
  const resultsContainer = document.getElementById("searchResults")
  const emptyState = document.getElementById("emptyState")
  const noResults = document.getElementById("noResults")
  const searchStats = document.getElementById("searchStats")
  const resultCount = document.getElementById("resultCount")

  emptyState.classList.add("hidden")

  if (results.length === 0) {
    resultsContainer.innerHTML = ""
    noResults.classList.remove("hidden")
    searchStats.classList.add("hidden")
    return
  }

  noResults.classList.add("hidden")
  searchStats.classList.remove("hidden")
  resultCount.textContent = results.length

  resultsContainer.innerHTML = results
    .map(
      (result) => `
    <div class="glass p-6 rounded-xl hover:border-purple-500 transition-all reveal active">
      <a href="${result.url}" class="block hover:no-underline group">
        <div class="flex items-start justify-between mb-3">
          <h3 class="text-2xl font-semibold group-hover:text-gradient">${highlightText(result.title, query)}</h3>
          <i class="fas fa-arrow-right text-muted group-hover:text-gradient transition-colors"></i>
        </div>
        
        <p class="text-muted mb-3">${highlightText(result.description, query)}</p>
        
        <div class="flex flex-wrap gap-2">
          ${result.matchedIn
            .map(
              (match) => `
            <span class="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
              Match in ${match}
            </span>
          `,
            )
            .join("")}
        </div>
        
        ${
          result.keywords.length > 0
            ? `
          <div class="mt-3 flex flex-wrap gap-2">
            ${result.keywords
              .slice(0, 5)
              .map(
                (keyword) => `
              <span class="text-xs px-2 py-1 rounded-full bg-white/5 text-muted border border-white/10">
                ${keyword}
              </span>
            `,
              )
              .join("")}
          </div>
        `
            : ""
        }
      </a>
    </div>
  `,
    )
    .join("")
}

// Highlight matching text
function highlightText(text, query) {
  if (!query) return text

  const regex = new RegExp(`(${query})`, "gi")
  return text.replace(regex, '<span class="text-gradient font-semibold">$1</span>')
}

// Show empty state
function showEmptyState() {
  document.getElementById("searchResults").innerHTML = ""
  document.getElementById("emptyState").classList.remove("hidden")
  document.getElementById("noResults").classList.add("hidden")
  document.getElementById("searchStats").classList.add("hidden")
}

// Initialize search
document.addEventListener("DOMContentLoaded", () => {
  loadSearchIndex()

  const searchInput = document.getElementById("searchInput")

  // Handle search input
  searchInput.addEventListener("input", (e) => {
    clearTimeout(searchTimeout)
    searchTimeout = setTimeout(() => {
      performSearch(e.target.value)
    }, 300)
  })

  // Handle Enter key
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      clearTimeout(searchTimeout)
      performSearch(e.target.value)
    }
  })

  // Focus on / key
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== searchInput) {
      e.preventDefault()
      searchInput.focus()
    }
  })

  // Quick search buttons
  document.querySelectorAll(".quick-search").forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-query")
      searchInput.value = query
      performSearch(query)
    })
  })

  // Check for query parameter
  const urlParams = new URLSearchParams(window.location.search)
  const query = urlParams.get("q")
  if (query) {
    searchInput.value = query
    performSearch(query)
  }
})
