import sys
import csv
import os
import hashlib
from bs4 import BeautifulSoup
from collections import deque
import logging

# === SETTING UP AND CONFIGURING LOGGER ===

def setup_logger(log_level=logging.INFO, log_file=None):
    logger = logging.getLogger('document_analyzer')
    
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# === ANALYZE DOCUMENT STRUCTURE ===

# This script analyzes the structure of XML/HTML documents and outputs data to CSV format. 
# Process flow:
# 1. Load the document and parse it using BeautifulSoup's XML parser
# 2. If XML parsing fails, fall back to HTML parser to obtain the root element
# 3. Build a tree structure and determine node types based on HTML tags and content structure
# 4. Identify sections, subsections, and tables:
#    - If semantic HTML tags (section, article, etc.) are present, use them for structure detection
#    - Otherwise, analyze different levels of the document tree to infer structural hierarchy
# 5. Convert identified tables to a structured format
# 6. Extract data to CSV with hierarchical section/subsection information preserved
# 7. Write the structured data to a CSV file with appropriate headers

def analyze_document_structure(file_path, csv_output_path=None, logger = None):
    
    logger = logger or logging.getLogger('document_analyzer')
    
    try:
        logger.info(f"Loading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.debug(f"Successfully loaded file with {len(content)} characters")
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        print(f"Error loading file: {str(e)}")
        sys.exit(1)
    
    logger.info("Parsing document with BeautifulSoup")
    soup = BeautifulSoup(content, 'lxml-xml')
    if not soup.find():
        logger.info("XML parsing failed, falling back to HTML parser")
        soup = BeautifulSoup(content, 'lxml')
    
    root = soup.find()
    if not root:
        print("Error: Could not find root element")
        sys.exit(1)
    
    
    logger.info("Building document tree structure")
    tree = build_tree(root)
    
    
    logger.info("Classifying nodes based on HTML tags")
    classify_nodes_respecting_html_tags(tree, logger)
    
    print("\n=== DOCUMENT STRUCTURE ===")
    print_tree(tree)
    
    
    logger.info("Processing tables in document")
    tables_dict = process_tables_in_document(tree, logger)
    
    
    logger.info("Extracting CSV data")
    csv_data = extract_csv_data(tree, tables_dict)
    
    if not csv_output_path:
        structure_hash = generate_structure_hash(csv_data)
        csv_output_path = f"document_structure_{structure_hash}.csv"
        logger.info(f"Generated output filename: {csv_output_path}")
    
    logger.info(f"Writing data to CSV: {csv_output_path}")
    write_to_csv(csv_data, csv_output_path, logger)
    
    return tree

# === GENERATING STRUCTURE HASH ===

# Implements a deterministic hashing algorithm (MD5) to create a unique document fingerprint
# Uses string joining operations to concatenate CSV headers into a single representative string
# The hash is specifically used to generate unique but reproducible filenames for output files

def generate_structure_hash(csv_data):
    headers_str = ",".join(csv_data["headers"])
    hash_obj = hashlib.md5(headers_str.encode())
    return hash_obj.hexdigest()[:8]

# === BUILDING TREE ===

# Implements a breadth-first search (BFS) traversal using a collections.deque data structure
# The BFS approach is specifically used to:
# Process document nodes level-by-level, ensuring parents are processed before children
# Build a complete hierarchical representation while maintaining accurate depth information
# Handle deeply nested documents efficiently without recursion stack limitations
# Uses a dictionary-based approach (sibling_maps) to track and associate nodes at the same hierarchical level
# The node structure uses nested dictionaries to store metadata about each element in the document
# Path notation employs a dot-separated string (e.g., "0.1.2") as a unique identifier system for nodes

def build_tree(root, logger = None):
    
    logger = logger or logging.getLogger('document_analyzer')

    if not hasattr(root, 'name') or not root.name:
        
        logger.warning("Root element has no name attribute")
        return None


    logger.debug(f"Building tree from root element: {root.name}")
    
    root_node = {
        "tag": root.name,
        "type": "root",
        "level": 0,
        "path": "0",
        "title": get_node_title(root),
        "content": get_direct_content(root),
        "content_length": len(get_direct_content(root)),
        "has_content": bool(get_direct_content(root).strip()),
        "children": [],
        "parent": None,
        "siblings": [],
        "element": root
    }
    
    queue = deque([(root, root_node, 0)])
    
    sibling_maps = {}
    
    # BFS Traversal for building a complete hierarchical node structure level by level.
    # It constructs a tree representation where each node contains metadata about its position, content, and relationships to other nodes. 
    # The queue ensures parent nodes are processed before their children, maintaining proper hierarchical relationships.
    
    while queue:
        element, parent_node, level = queue.popleft()
        
        child_index = 0
        current_level_children = []
        
        for child in element.find_all(recursive=False):
            if not hasattr(child, 'name') or not child.name:
                continue
                
            if child.name in ['br', 'hr', 'img', 'span', 'a', 'em', 'strong', 'i', 'b', 'code']:
                continue
            
            child_path = f"{parent_node['path']}.{child_index}"
            child_content = get_direct_content(child)
            
            initial_type = "element"
            if child.name.lower() == 'section':
                initial_type = "section"
            elif child.name.lower() == 'subsection':
                initial_type = "subsection"
            
            child_node = {
                "tag": child.name,
                "type": initial_type,
                "level": level + 1,
                "path": child_path,
                "title": get_node_title(child),
                "content": child_content,
                "content_length": len(child_content),
                "has_content": bool(child_content.strip()),
                "children": [],
                "parent": parent_node,
                "siblings": [],
                "element": child
            }
            
            current_level_children.append(child_node)
            
            parent_node["children"].append(child_node)
            
            queue.append((child, child_node, level + 1))
            child_index += 1
        
        if current_level_children:
            sibling_key = (parent_node["path"], level + 1)
            sibling_maps[sibling_key] = current_level_children # Map to track siblings for relationship analysis
    
    for sibling_group in sibling_maps.values():
        for node in sibling_group:
            node["siblings"] = sibling_group
    
    
    logger.debug(f"Tree built with {len(sibling_maps)} sibling groups")
    
    return root_node


# === GETTING NODE'S TITLE ===

# Uses a multi-strategy pattern matching algorithm with priority fallbacks:
# Direct attribute access for "title" attribute
# Child element search for heading tags
# ID attribute parsing with string transformations
# Text content extraction with length limiting
# String slicing operations are used for content preview generation
# Each strategy is attempted in sequence until a suitable title is found

def get_node_title(element):
    if 'title' in element.attrs: # Check for explicit title attribute first
        return element.attrs['title']
    
    for heading_tag in ['title', 'h1', 'h2', 'h3', 'h4', 'heading']:
        heading = element.find(heading_tag, recursive=False)
        if heading:
            return heading.get_text(strip=True)
    
    if 'id' in element.attrs:
        return element.attrs['id'].replace('-', ' ').replace('_', ' ')
    
    direct_text = get_direct_content(element).strip()
    if direct_text:
        preview = direct_text[:50] + ('...' if len(direct_text) > 50 else '')
        return f"{element.name}: {preview}"
    
    return element.name


# === GET DIRECT CONTENT ===

# Uses list comprehension with type checking to filter and concatenate only direct text node children

def get_direct_content(element):
    return "".join(content for content in element.contents 
                  if isinstance(content, str))

# === GET FULL CONTENT ===

# Applies BeautifulSoup's recursive text extraction algorithm with whitespace normalization

def get_full_content(element):
    return element.get_text(strip=True)

# === CLASSIFYING NODES BASED ON HTML TAGS ===

# Implements a three-phase classification pipeline:
# First BFS traversal: Uses deque to collect unclassified nodes throughout the tree
# Second phase: Applies heuristic classification to the collected nodes
# Third BFS traversal: Propagates section/subsection relationships
# Fourth BFS traversal: Finalizes remaining element types
# Each BFS traversal completely processes the tree using a queue-based approach for efficient level order processing
# The function specifically avoids reclassifying nodes that already have semantic types

def classify_nodes_respecting_html_tags(tree, logger = None):
    
    logger = logger or logging.getLogger('document_analyzer')
    logger.debug("Starting node classification")
    
    nodes_needing_classification = []
    
    queue = deque([tree])
    
    # BFS Traversal for collecting all nodes that need type classification, excluding those already classified as sections or subsections. 
    # It populates a list of nodes requiring heuristic-based classification while preserving the existing semantic structure of the document.
    while queue:
        node = queue.popleft()
        
        if node["level"] > 0:
            if node["type"] not in ["section", "subsection"]:
                nodes_needing_classification.append(node)
        
        for child in node["children"]:
            queue.append(child)
    
    classify_nodes_by_heuristics(tree, nodes_needing_classification)
    logger.debug(f"Classified {len(nodes_needing_classification)} nodes by heuristics")
    
    
    queue = deque([tree])
    
    # BFS Traversal for propagating section relationships downward through the hierarchy. 
    # It ensures that direct children of sections are properly classified as subsections, establishing correct parent-child structural relationships.
    
    while queue:
        node = queue.popleft()
        
        if node["type"] == "section":
            for child in node["children"]:
                if child["type"] not in ["section", "subsection"]:
                    child["type"] = "subsection"
        
        for child in node["children"]:
            queue.append(child)
    
    queue = deque([tree])
    
    # BFS Traversal for finalizing the classification of remaining elements based on their content and children. 
    # It distinguishes between containers (elements with children) and content elements (elements with text but no children).
    
    while queue:
        node = queue.popleft()
        
        if node["type"] == "element":
            if node["children"]:
                node["type"] = "container"
            elif node["has_content"]:
                node["type"] = "content"
        
        for child in node["children"]:
            queue.append(child)
    
    
    logger.debug("Completed node classification")
            
# === CLASSIFYING NODES BASED ON HEURISTICS ===

# Uses a dictionary-based grouping algorithm to organize nodes by their parent path
# Implements a frequency analysis approach to identify potential section patterns
# Uses a threshold-based decision system (nodes > 1) to determine section classification
# Delegates to specialized classification for nodes that don't match group patterns

def classify_nodes_by_heuristics(tree, nodes_to_classify):
    nodes_by_parent = {}
    
    for node in nodes_to_classify:
        parent_path = node["parent"]["path"]
        
        if parent_path not in nodes_by_parent:
            nodes_by_parent[parent_path] = []
        
        nodes_by_parent[parent_path].append(node)
    
    for parent_path, nodes in nodes_by_parent.items():
        nodes_with_content_children = []
        
        for node in nodes:
            has_children_with_content = any(child["has_content"] for child in node["children"])
            if has_children_with_content:
                nodes_with_content_children.append(node)
        
        if len(nodes_with_content_children) > 1: # classify as section if multiple nodes have content children
            for node in nodes:
                if node["type"] not in ["section", "subsection", "root"]:
                    node["type"] = "section"
    
    for node in nodes_to_classify:
        if node["type"] not in ["section", "subsection", "root"]:
            classify_by_characteristics(node)
            
# === CLASSIFYING NODES BY CHARACTERISTICS ===

# Implements a decision tree algorithm for node type classification
# Uses tag name matching against predefined sets of HTML elements for initial classification
# Applies regular expression-like pattern matching for content structure analysis (e.g., colon-separated fields)
# Has specific content length thresholds (e.g., < 100 chars) for certain classification decisions
# Contains a multi-tier fallback system that ensures every node receives a classification

def classify_by_characteristics(node):
    tag = node["tag"].lower()
    
    if node["type"] in ["section", "subsection"]:
        return
    
    if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'heading', 'title']:
        node["type"] = "heading"
    elif tag in ['p', 'div'] and node["has_content"]:
        node["type"] = "content"
    elif tag in ['li', 'dt', 'dd']:  # Identify List Items by tag
        node["type"] = "list-item"
    elif tag in ['tr']:
        node["type"] = "table-row"
    elif tag in ['td', 'th']:
        node["type"] = "table-cell"
    elif tag in ['ul', 'ol', 'dl', 'menu']:
        node["type"] = "list"
    elif tag in ['table', 'thead', 'tbody', 'tfoot']:
        node["type"] = "table"
    elif tag in ['form', 'fieldset']:
        node["type"] = "form"
    elif node["has_content"] and len(node["content"].strip()) < 100 and ":" in node["content"]:
        field_parts = node["content"].split(":", 1)
        if len(field_parts) == 2 and field_parts[0].strip() and field_parts[1].strip():
            node["type"] = "field"
    elif node["children"] and not node["has_content"]:
        node["type"] = "container"
    elif node["has_content"]:
        node["type"] = "content"
    else:
        node["type"] = "element"

# === TREE VISUALIZATION === 

# Implements a recursive depth-first traversal specialized for visual representation
# Uses string formatting and ASCII box-drawing characters to create a visual tree structure
# Implements branch continuation logic to show relationships across multiple levels
# The recursion tracks both depth (via indent parameter) and sibling position (via is_last parameter)
# String concatenation is used to build the properly indented visual representation

def print_tree(node, indent="", is_last=True):
    if not node:
        return
    
    branch = "└── " if is_last else "├── "
    
    node_text = f"{node['type']}: {node['title']}"
    
    if node.get("has_content"):
        node_text += f" [content: {node['content_length']} chars]"
    
    print(f"{indent}{branch}{node_text}")
    
    children = node["children"]
    child_indent = indent + ("    " if is_last else "│   ")
    
    for i, child in enumerate(children):
        print_tree(child, child_indent, i == len(children) - 1)
        

# === PROCESSING TABLES IN TREE ===

# Implements a BFS traversal to access all nodes in the document tree
# For each node, applies BeautifulSoup's recursive search algorithm to find all table elements
# Uses a dictionary data structure to map table identifiers to their processed content
# The BFS approach ensures tables are discovered in hierarchical order for proper contextual placement

def process_tables_in_document(tree, logger = None):
    
    logger = logger or logging.getLogger('document_analyzer')
        
    tables_dict = {}
    try:
        queue = deque([tree])
        
        # BFS Traversal for locating and processing all table elements in the document tree. 
        # It extracts table data and builds a dictionary mapping table identifiers to their formatted content while maintaining the hierarchical context of each table.
      
        table_index = 1
        
        while queue:
            node = queue.popleft()
            
            table_elements = node["element"].find_all('table', recursive=True)
            
            for table_element in table_elements:
                markup = process_table_from_element(table_element)
                if markup:
                    parent_path = find_parent_path_for_element(tree, table_element)
                    if parent_path:
                        key = f"{parent_path}_table_{table_index}"
                        tables_dict[key] = markup
                        table_index += 1
            
            for child in node["children"]:
                queue.append(child)
                
        logger.info(f"Processed {table_index-1} tables in document")
        return tables_dict
    except Exception as e :

        logger.error(f"Error processing tables: {str(e)}")
        print(f"Error : {str(e)}")
        

# === FINDING PARENT OF NODE ===

# Implements a BFS traversal using a deque for efficient search operations
# The search algorithm checks for both direct element matches and descendant relationships
# Returns the path immediately upon finding a match, optimizing for early termination
# The BFS approach ensures the closest ancestor is found when multiple containers might match

def find_parent_path_for_element(tree, element):
    queue = deque([tree])
    
    # BFS Traversal for locating the parent node of a specific element in the tree.
    # It searches through the tree to find the closest container that either matches the element directly or contains it as a descendant.
    
    while queue:
        node = queue.popleft()
        
        if node["element"] == element or node["element"].find(element):
            return node["path"]
        
        for child in node["children"]:
            queue.append(child)
    
    return None

# === PROCESSING TABLE FROM ELEMENTS ===

# Implements a table parsing algorithm specialized for HTML table structures
# Uses a list-based approach to collect rows and cells from the HTML structure
# Applies a dynamic column width calculation algorithm to ensure proper alignment
# Implements string padding and justification to create a consistently formatted text table
# Uses nested list operations to process row and cell data with proper formatting

def process_table_from_element(table_element):
    rows = []
    header_row = []
    
    tr_elements = table_element.find_all('tr', recursive=True)
    
    if not tr_elements:
        return None
        
    for tr in tr_elements:
        row = []
        cells = tr.find_all(['th', 'td'], recursive=False)
        for cell in cells:
            cell_content = cell.get_text(strip=True)
            row.append(cell_content)
        
        if not rows and tr.find('th', recursive=False):
            header_row = row
        else:
            rows.append(row)
    
    if not header_row and rows:
        header_row = rows[0]
        rows = rows[1:]
    
    if not header_row:
        return None
    
    col_widths = [len(str(col)) for col in header_row] # Calculate column widths to ensure proper alignment 
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths): 
                col_widths[i] = max(col_widths[i], len(str(cell))) # Adjust column widths based on content
    
    markup = []
    
    header = " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(header_row))
    markup.append(header)
    
    separator = " | ".join("-" * width for width in col_widths)
    markup.append(separator)
    
    for row in rows:
        row_str = " | ".join(str(cell).ljust(col_widths[i]) if i < len(col_widths) else "" 
                            for i, cell in enumerate(row))
        markup.append(row_str)
    
    return "\n".join(markup)

# === GETTING HIERARCHIAL NAME ===

# Implements a bottom-up traversal algorithm starting from the current node
# Uses recursive parent access to build a complete hierarchical path
# Employs a filtering system to include only structural elements in the path
# Implements list operations (append, reverse) to construct the path in correct order
# Uses string joining with dot notation to create the final hierarchical name

def get_full_hierarchy_name(node):
    hierarchy = []
    
    current = node
    while current:
        if current["type"] in ["root", "container", "section", "subsection"]:
            node_name = extract_clean_name(current["title"])
            hierarchy.append((current["type"], node_name))
        
        current = current.get("parent")
    
    hierarchy.reverse()
    
    full_name_parts = []
    for node_type, node_name in hierarchy:
        full_name_parts.append(node_name)
    
    return ".".join(full_name_parts)

# === EXTRACTING CSV DATA ===

# Implements a multi-phase data extraction pipeline with specialized processors:
# Field processor: Extracts explicit field-value pairs
# Section processor: Handles standard sections and their content
# Repeated section processor: Processes recurring structural patterns
# Table processor: Incorporates tables with proper hierarchical context
# Uses a dictionary data structure with parallel arrays for headers and values
# The pipeline design ensures each type of content is processed with appropriate context

def extract_csv_data(tree, tables_dict):
    data = {
        'headers': [],
        'values': []
    }
    
    process_fields(tree, data)
    
    process_sections(tree, data, tables_dict)
    
    process_repeated_sections(tree, data, tables_dict)
    
    for table_key, table_markup in tables_dict.items():  # Process remaining tables with hierarchical context
        if "_table_" in table_key:
            path_part = table_key.split('_table_')[0]
            table_num = table_key.split('_table_')[1]
            
            node = find_node_by_path(tree, path_part)
            if node:
                field_name = f"{get_full_hierarchy_name(node)}.Table_{table_num}"
            else:
                field_name = f"Table_{table_num}"
            
            if field_name not in data["headers"]:
                data["headers"].append(field_name)
                data["values"].append(table_markup)
    
    return data

# === PROCESSING FIELDS === 

# Implements a BFS traversal to locate all field-type nodes in the document
# Uses string parsing with split operations to extract key-value pairs from content
# Applies hierarchical naming algorithms to create fully qualified field names
# The BFS approach ensures fields are discovered in hierarchical order for proper context

def process_fields(tree, data):
    
    fields = []
    
    queue = deque([tree])
    
    # BFS Traversal for identifying all field-type nodes in the document.
    # It extracts key-value pairs from nodes with a colon-separated content pattern, capturing form fields and other structured data.
    
    while queue:
        node = queue.popleft()
        
        if node["type"] == "field":
            fields.append(node)
        
        for child in node["children"]:
            queue.append(child)
    
    for field in fields:
        if field["has_content"] and ":" in field["content"]:
            parts = field["content"].split(":", 1)
            if len(parts) == 2:
                field_name = parts[0].strip()
                field_value = parts[1].strip()
                parent_path = get_full_hierarchy_name(field["parent"])
                full_field_name = f"{parent_path}.{field_name}" if parent_path else field_name # Create fully qualified field name with hierarchy
                
                data["headers"].append(full_field_name)
                data["values"].append(field_value)
                
# === FINDING NODE BY PATH === 

# Implements a BFS traversal optimized for path matching
# Uses direct string comparison for efficient path identification
# Early termination optimization returns immediately upon finding a match
# Complete tree traversal capability ensures any node can be found regardless of depth

def find_node_by_path(tree, path):
    if tree["path"] == path:
        return tree
    
    queue = deque([tree])
    
    # BFS Traversal for locating a node by its path identifier. 
    # It traverses the tree comparing path strings until it finds an exact match, allowing for direct access to specific nodes
    
    while queue:
        node = queue.popleft()
        
        if node["path"] == path:
            return node
        
        for child in node["children"]:
            queue.append(child)
    
    return None

# === PROCESSING SECTIONS ===

# Uses BFS traversal to collect all section nodes throughout the document
# Implements a filtering algorithm to identify and skip repeated sections
# Applies string parsing operations to extract section content with special handling for colon patterns
# Uses path prefix matching to associate tables with their parent sections
# Employs a deletion strategy to mark processed tables and avoid duplication

def process_sections(tree, data, tables_dict):
    sections = []
    
    queue = deque([tree])
    
    # BFS Traversal for collecting all section nodes throughout the document. 
    # It identifies sections for content extraction while filtering out repeated patterns, preparing them for structured data output.
    
    while queue:
        node = queue.popleft()
        
        if node["type"] == "section":
            sections.append(node)
        
        for child in node["children"]:
            queue.append(child)
    
    for section in sections:
        section_name = get_full_hierarchy_name(section)
        section_path = section["path"]
        
        if has_repeated_instances(section, sections):  # Skip sections that are part of repeated patterns
            continue
            
        if section["has_content"] and section["content"].strip():
            value = section["content"].strip()
            if ":" in value:
                parts = value.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    value = parts[1].strip()
            
            data["headers"].append(section_name)
            data["values"].append(value)
        
        for child in section["children"]:
            if child["type"] == "subsection":
                process_subsection(section_name, child, data, tables_dict)
        
        for table_key in list(tables_dict.keys()):
            if table_key.startswith(section_path + "_table_"):
                table_num = table_key.split('_table_')[1]
                field_name = f"{section_name}.Table_{table_num}"
                
                data["headers"].append(field_name)
                data["values"].append(tables_dict[table_key])
                
                del tables_dict[table_key]
                
                
# PROCESSING SUBSECTIONS

# Implements focused content extraction specialized for subsection nodes
# Uses recursive content extraction when direct content is unavailable
# Applies hierarchical naming algorithms to maintain proper section-subsection relationships
# Uses string operations for content cleaning and normalization
# Implements path prefix matching to associate tables with their subsections

def process_subsection(section_name, subsection, data, tables_dict):
    subsection_name = extract_clean_name(subsection["title"])
    subsection_path = subsection["path"]
    
    if not subsection["has_content"] and not subsection["children"]:
        return
    
    value = extract_value_from_node(subsection)
    
    if value:
        field_name = f"{section_name}.{subsection_name}"
        
        data["headers"].append(field_name)
        data["values"].append(value)
    
    for child in subsection["children"]:
        if child["type"] == "field" and child["has_content"] and ":" in child["content"]:
            parts = child["content"].split(":", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                field_name = f"{section_name}.{subsection_name}.{parts[0].strip()}"
                
                data["headers"].append(field_name)
                data["values"].append(parts[1].strip())
    
    for table_key in list(tables_dict.keys()):
        if table_key.startswith(subsection_path + "_table_"):
            table_num = table_key.split('_table_')[1]
            field_name = f"{section_name}.{subsection_name}.Table_{table_num}"
            
            data["headers"].append(field_name)
            data["values"].append(tables_dict[table_key])
            
            del tables_dict[table_key]
            
# === PROCESSING REPEATED SECTIONS ===

# Implements a frequency analysis algorithm using a dictionary-based counting approach
# Uses BFS traversal to locate all section nodes in the document
# Applies name normalization to identify conceptually identical sections
# Implements a threshold-based detection system (count > 1) for repeated patterns
# Groups related sections for specialized processing with indexed naming

def process_repeated_sections(tree, data, tables_dict):
    section_counts = {}
    
    queue = deque([tree])
    
    # BFS Traversal for performing frequency analysis on section types. 
    # It identifies repeated section patterns by counting nodes with the same normalized title, enabling specialized processing for recurring document structures.
    
    while queue:
        node = queue.popleft()
        
        if node["type"] == "section":
            section_name = extract_clean_name(node["title"])
            full_section_name = get_full_hierarchy_name(node)
            
            if section_name not in section_counts:
                section_counts[section_name] = {'nodes': [], 'full_name': full_section_name}
            
            section_counts[section_name]['nodes'].append(node)
        
        for child in node["children"]:
            queue.append(child)
    
    for section_name, section_info in section_counts.items():
        if len(section_info['nodes']) > 1:  # Process sections that appear multiple times
            process_repeated_section_group(section_info['full_name'], section_info['nodes'], data, tables_dict)


# === PROCESSING REPEATED SECTION GROUP ===

# Implements a template-based processing system using the first instance as a structural model
# Uses indexed notation (e.g., "Section[1]") to distinguish between repeated instances
# Applies consistent field extraction across all instances using the template structure
# Uses path prefix matching to associate tables with the correct instances
# Implements a hierarchical naming scheme that preserves both repetition and structure

def process_repeated_section_group(full_section_name, sections, data, tables_dict):
    template_section = sections[0]
    
    subsection_names = []
    for child in template_section["children"]:
        if child["type"] == "subsection":
            subsection_name = extract_clean_name(child["title"])
            if subsection_name not in subsection_names:
                subsection_names.append(subsection_name)
    
    for i, section in enumerate(sections):
        section_path = section["path"]
        
        section_index = i + 1
        
        for subsection_name in subsection_names:
            for child in section["children"]:
                if child["type"] == "subsection" and extract_clean_name(child["title"]) == subsection_name:
                    value = extract_value_from_node(child)
                    
                    if value:
                        field_name = f"{full_section_name}[{section_index}].{subsection_name}"
                        
                        data["headers"].append(field_name)
                        data["values"].append(value)
                    
                    for subchild in child["children"]:
                        if subchild["type"] == "field" and subchild["has_content"] and ":" in subchild["content"]:
                            parts = subchild["content"].split(":", 1)
                            if len(parts) == 2:
                                field_key = parts[0].strip()
                                field_value = parts[1].strip()
                                
                                field_name = f"{full_section_name}[{section_index}].{subsection_name}.{field_key}"
                                
                                data["headers"].append(field_name)
                                data["values"].append(field_value)
                    
                    subsection_path = child["path"]
                    for table_key in list(tables_dict.keys()):
                        if table_key.startswith(subsection_path + "_table_"):
                            table_num = table_key.split('_table_')[1]
                            table_field_name = f"{full_section_name}[{section_index}].{subsection_name}.Table_{table_num}"
                            
                            data["headers"].append(table_field_name)
                            data["values"].append(tables_dict[table_key])
                            
                            del tables_dict[table_key]
                    
                    break
        
        for table_key in list(tables_dict.keys()):
            if table_key.startswith(section_path + "_table_"):
                table_num = table_key.split('_table_')[1]
                field_name = f"{full_section_name}[{section_index}].Table_{table_num}"
                
                data["headers"].append(field_name)
                data["values"].append(tables_dict[table_key])
                
                del tables_dict[table_key]
                
# === EXTRACTING CLEAN NAME === 

# Implements a string parsing algorithm specialized for title cleaning
# Uses string splitting and conditional logic to handle various title patterns
# Applies specific handling for "section:" and "subsection:" prefixes
# Uses multiple split operations to handle complex title structures
# Focuses on extracting the most meaningful part of compound titles

def extract_clean_name(title):
    if ':' in title: # Handle titles with prefixes like "section:" or "subsection:"
        parts = title.split(':', 1)
        if parts[0].strip().lower() in ["section", "subsection", "content"]:
            title = parts[1].strip() # Remove prefix if present
    
    if ':' in title:
        title = title.split(':', 1)[0].strip()
    
    return title

# === EXTRACTING VALUE FROM NODE ===

# Implements a multi-strategy content extraction algorithm:
# Direct content extraction with pattern recognition
# Child content collection for nodes without direct content
# Recursive processing for nested content structures
# Uses string operations for pattern matching and content cleaning
# Implements list joining with semicolon separators for multiple content pieces
# Contains special case handling for specific field types like "Abbreviation" and "LongName"

def extract_value_from_node(node):
    if node["has_content"] and node["content"].strip():
        content = node["content"].strip()
        
        if ":" in content: # Check for field-value pattern in content
            parts = content.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip() # Return value part after colon
        
        return content
    
    child_values = []
    
    for child in node["children"]:
        if child["type"] == "field" and child["has_content"]:
            # Skip processing fields here as they will be handled separately
            continue
        elif child["has_content"]:
            content = child["content"].strip()
            
            if ":" in content:
                parts = content.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    child_type = child["type"]
                    if child_type == "content" and parts[0].strip() in ["Abbreviation", "LongName"]:
                        child_values.append(parts[1].strip())
                    else:
                        child_values.append(parts[1].strip())
                    continue
            
            child_values.append(content)
        elif len(child["children"]) > 0:
            nested_value = extract_value_from_node(child)
            if nested_value:
                child_values.append(nested_value)
    
    if child_values:
        return "; ".join(child_values)
    
    return None

# === CHECKING REPEATED INSTANCES ===

# Implements a frequency counting algorithm specific to section titles
# Uses list comprehension with conditional filtering to count matching sections
# Applies name normalization to identify conceptually identical sections
# Uses a threshold-based approach (count > 1) to identify repeated patterns

def has_repeated_instances(section, all_sections):
    section_name = extract_clean_name(section["title"])
    count = sum(1 for s in all_sections if extract_clean_name(s["title"]) == section_name)
    return count > 1

# === WRITING TO CSV FILE ===

# Implements CSV file writing using Python's built-in csv module
# Uses exception handling to gracefully manage file operation errors
# Writes data in a single row format with headers followed by values
# Uses proper UTF-8 encoding and newline handling for cross-platform compatibility

def write_to_csv(csv_data, output_path, logger = None):
    
    logger = logger or logging.getLogger('document_analyzer')
    
    try:
        logger.info(f"Writing {len(csv_data['headers'])} columns to CSV")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(csv_data["headers"])
            
            writer.writerow(csv_data["values"])
        
        print(f"\nCSV data written to {output_path}")
        
        logger.info(f"CSV data successfully written to {output_path}")
    except Exception as e:
        logger.error(f"Error writing CSV file: {str(e)}")
        print(f"Error writing CSV file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_file> [output_csv_path] [log_file_path] [log_level]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    csv_output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    log_file = sys.argv[3] if len(sys.argv) > 3 else None
    log_level_arg = sys.argv[4].upper() if len(sys.argv) > 4 else "INFO"
    log_level = getattr(logging, log_level_arg, logging.INFO)
    logger = setup_logger(log_level, log_file)
    logger.info(f"Starting document analysis on: {file_path}")
    
    analyze_document_structure(file_path, csv_output_path, logger)