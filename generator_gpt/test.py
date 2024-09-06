
from IPython.display import display, Markdown
import markdown


table = [
            [
                "year ended december 31 dollars in millions",
                "2009",
                "2008"
            ],
            [
                "net interest income",
                "$ 9083",
                "$ 3854"
            ],
            [
                "net interest margin",
                "3.82% ( 3.82 % )",
                "3.37% ( 3.37 % )"
            ]
        ]


def convert_to_markdown(table):
    # Generate the header line
    header = "| " + " | ".join(table[0]) + " |"
    # Generate the separator line
    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
    # Initialize the markdown table with header and separator
    markdown_table = [header, separator]
    
    # Loop over the table skipping the first row (since it's the header)
    for row in table[1:]:
        # Generate each row line
        row_line = "| " + " | ".join(row) + " |"
        markdown_table.append(row_line)
    
    return "\n".join(markdown_table)

def markdown_to_html(markdown_text):
    # Convert markdown text to HTML
    html_output = markdown.markdown(markdown_text, extensions=['tables'])
    return html_output


markdown_table = convert_to_markdown(table)
html_table = markdown_to_html(markdown_table)

print(markdown_table)

print(html_table)