#!/usr/bin/env python3
import mwclient
import json

WIKI_SETTINGS = {
    'host': 'solis-graphql.sl.cloud9.ibm.com',
    'path': '/wiki/',
    'username': "your_username",
    'password': "your_password"
}

def load_mapping_file():
    try:
        json_file = "mapping_output.json"
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file}': {e}")
        return None

def create_infobox(infobox_data):
    """Generate infobox wikitext from data"""
    if not infobox_data:
        return ""
    
    infobox_lines = [f"{{{{Infobox {infobox_data.get('type', 'Software')}"]
    
    for key, value in infobox_data.get('fields', {}).items():
        infobox_lines.append(f"| {key} = {value}")
    
    infobox_lines.append("}}")
    return "\n".join(infobox_lines)

def create_page_content(page_data):
    """Generate complete page content from JSON data"""
    content_parts = []
    
    # Add infobox if present
    if 'infobox' in page_data:
        content_parts.append(create_infobox(page_data['infobox']))
        content_parts.append("")
    
    # Add main title
    title = page_data.get('title', 'Untitled')
    content_parts.append(f"= {title} =")
    content_parts.append("")
    
    # Add introduction
    intro = page_data.get('introduction', f"'''{title}''' needs description.")
    content_parts.append(intro)
    content_parts.append("")
    
    # Add sections
    for section in page_data.get('sections', []):
        section_title = section.get('title', 'Section')
        section_content = section.get('content', "''Content to be added''")
        
        content_parts.append(f"== {section_title} ==")
        content_parts.append(section_content)
        content_parts.append("")
    
    # Add external links if present
    if 'external_links' in page_data and page_data['external_links']:
        content_parts.append("== External Links ==")
        for link in page_data['external_links']:
            content_parts.append(f"* [{link['url']} {link['title']}]")
        content_parts.append("")
    
    # Add categories
    for category in page_data.get('categories', []):
        content_parts.append(f"[[Category:{category}]]")
    
    return "\n".join(content_parts)

def update_main_page(site, link_data):
    """Update main page with links to new pages"""
    try:
        main_page = site.pages['Main Page']
        current_content = main_page.text()
        
        section_title = link_data.get('section', 'New Pages')
        link_text = link_data.get('text', link_data['page_title'])
        description = link_data.get('description', '')
        
        link_line = f"* [[{link_data['page_title']}]] - {description}" if description else f"* [[{link_data['page_title']}]]"
        
        # Check if link already exists
        if link_data['page_title'] not in current_content:
            if f"== {section_title} ==" in current_content:
                # Add to existing section
                updated_content = current_content.replace(
                    f"== {section_title} ==",
                    f"== {section_title} ==\n{link_line}"
                )
            else:
                # Create new section
                new_section = f"\n== {section_title} ==\n{link_line}\n"
                updated_content = current_content + new_section
            
            main_page.edit(updated_content, f'Added link to {link_data["page_title"]} page')
            print(f"Main Page updated with link to {link_data['page_title']}!")
        else:
            print(f"Link to {link_data['page_title']} already exists on Main Page")
            
    except Exception as e:
        print(f"Error updating Main Page: {e}")

def create_wiki_pages():
    """Main function to create wiki pages from mapping file"""
    
    mapping_data = load_mapping_file()
    if not mapping_data:
        return 0
    
    try:
        site = mwclient.Site(WIKI_SETTINGS['host'], path=WIKI_SETTINGS['path'], scheme='http')
        print(f"Connected to http://{WIKI_SETTINGS['host']}{WIKI_SETTINGS['path']}")
    except Exception as e:
        print(f"Error connecting to wiki: {e}")
        return 0
    
    if WIKI_SETTINGS['username'] and WIKI_SETTINGS['password']:
        try:
            site.login(WIKI_SETTINGS['username'], WIKI_SETTINGS['password'])
            print("Logged in successfully")
        except Exception as e:
            print(f"Login failed: {e}")
            print("Continuing without login (anonymous editing)")
    
    # Process each page in the mapping?
    # need to think how this mapping_output.json file will look like
    pages_created = 0
    for page_data in mapping_data.get('pages', []):
        page_title = page_data.get('title', 'Untitled')
        print(f"\nCreating page: {page_title}")
        
        try:
            # Generate page content
            page_content = create_page_content(page_data)
            
            # Create the page
            page = site.pages[page_title]
            edit_summary = page_data.get('edit_summary', f'Created {page_title} page from mapping file')
            page.edit(page_content, edit_summary)
            print(f"Page '{page_title}' created successfully!")
            pages_created += 1
            
            # Update main page if specified
            if 'main_page_link' in page_data:
                link_data = page_data['main_page_link']
                link_data['page_title'] = page_title
                update_main_page(site, link_data)
                
        except Exception as e:
            print(f"Error creating page '{page_title}': {e}")
    
    return pages_created

def main():
    
    
    pages_created = create_wiki_pages()
    
    if pages_created > 0:
        print(f"Successfully created {pages_created} pages!")
    else:
        print("Errors in page creattion")

if __name__ == "__main__":
    main()
