document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("search-form");
    const queryTypeDropdown = document.getElementById("query-type");
    const lamInput = document.getElementById("lam-input");

    form.addEventListener("submit", async function (event) {
        event.preventDefault(); 

        const queryType = queryTypeDropdown.value; 
        const formData = new FormData(form);

        if (queryType === "hybrid") {
            formData.append("lam", lamInput.value);
        }

        let endpoint = "/";
        if (queryType === "text") {
            endpoint = "/text"; 
        } else if (queryType === "image") {
            endpoint = "/upload"; 
        } else if (queryType === "hybrid") {
            endpoint = "/search"; 
        }

        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        const resultDiv = document.getElementById("result");

        if (result.error) {
            resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
        } else {
            resultDiv.innerHTML = `<p>Query Type: ${queryType}</p>`;
            
            const resultsList = document.createElement("ul");
            result.top_results.forEach((item, index) => {
                const listItem = document.createElement("li");
                listItem.innerHTML = `
                    <p>Rank ${index + 1}:</p>
                    <img src="/images/${item.file_name}" alt="Image ${index + 1}" style="max-width: 300px;">
                    <p>Similarity Score: ${item.similarity.toFixed(2)}</p>
                `;
                resultsList.appendChild(listItem);
            });

            resultDiv.appendChild(resultsList);
        }
    });
});
