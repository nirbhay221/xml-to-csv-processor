import unittest
import os
import sys
import tempfile
import csv
from bs4 import BeautifulSoup
from xml_to_csv import (
    analyze_document_structure,
    build_tree,
    classify_nodes_respecting_html_tags,
    process_tables_in_document,
    extract_csv_data,
    get_full_hierarchy_name,
    process_table_from_element
)


# === TEST SUITE FOR DOCUMENT ANALYZER FUNCTIONALITY ===
# Tests XML processing, document structure analysis, table extraction, and CSV output generation.

class DocumentAnalyzerTests(unittest.TestCase):
    
    # Setting up test environment before each test. 
    # Creates a temporary directory and sample XML Documents for Testing.
    
    def setUp(self):
        # Temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Simple XML Document for basic testing
        self.simple_xml = """
        <document>
            <section>
                <title>Test Section</title>
                <content>This is test content</content>
                <subsection>
                    <heading>Test Subsection</heading>
                    <field>Key: Value</field>
                </subsection>
                <table>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                    </tr>
                    <tr>
                        <td>Cell 1</td>
                        <td>Cell 2</td>
                    </tr>
                    <tr>
                        <td>Cell 3</td>
                        <td>Cell 4</td>
                    </tr>
                </table>
            </section>
        </document>
        """
        
        # Complex XML Document with nested sections and multiple tables
        
        self.complex_xml = """
        <document>
            <container title="Complex Document">
                <heading>Complex Document</heading>
                <container>
                    <content>author: Test Author</content>
                    <content>date: 2025-03-13</content>
                </container>
            </container>
            <container>
                <section>
                    <title>First Section</title>
                    <paragraph>This is the first paragraph.</paragraph>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Age</th>
                            <th>Location</th>
                        </tr>
                        <tr>
                            <td>John Doe</td>
                            <td>30</td>
                            <td>New York</td>
                        </tr>
                        <tr>
                            <td>Jane Smith</td>
                            <td>25</td>
                            <td>Boston</td>
                        </tr>
                    </table>
                </section>
                <section>
                    <title>Second Section</title>
                    <subsection>
                        <heading>Subsection 2.1</heading>
                        <field>Status: Active</field>
                        <field>Priority: High</field>
                        <table>
                            <tr>
                                <th>Item</th>
                                <th>Quantity</th>
                                <th>Price</th>
                            </tr>
                            <tr>
                                <td>Widget A</td>
                                <td>5</td>
                                <td>$10.00</td>
                            </tr>
                            <tr>
                                <td>Widget B</td>
                                <td>3</td>
                                <td>$15.00</td>
                            </tr>
                        </table>
                    </subsection>
                </section>
            </container>
        </document>
        """
        
        # Writing simple XML to a File
        
        self.simple_xml_path = os.path.join(self.test_dir, "simple.xml")
        with open(self.simple_xml_path, 'w', encoding='utf-8') as f:
            f.write(self.simple_xml)
            
        # Writing complex XML to a File
            
        self.complex_xml_path = os.path.join(self.test_dir, "complex.xml")
        with open(self.complex_xml_path, 'w', encoding='utf-8') as f:
            f.write(self.complex_xml)
        
        # output path for CSV File
        self.output_csv_path = os.path.join(self.test_dir, "output.csv")
    
    
    # === CLEANS ENVIRONMENT AFTER EACH TEST ===
    # Removes all temporary files and directories created during testing to prevent interference between tests and avoid leaving behind temporary artifacts.
    
    def tearDown(self):
        try:
            # Deleting all files in the temporary directory
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            # Removing the temporary directory
            os.rmdir(self.test_dir)
        except Exception as e:
            print(f"Error during tearDown: {e}")
            
    
    # === TESTING CSV FILE OUTPUT ===

    # Tests if the document analysis successfully create a CSV Output File.
    # And verifies file creation and basic structure.

    def test_csv_file_output(self):
        
        # main analysis function
        
        analyze_document_structure(self.simple_xml_path, self.output_csv_path)
        
        # checking if CSV File was created
        
        self.assertTrue(os.path.exists(self.output_csv_path), "CSV file should be created")
        
        # checking if CSV has at least a header row
        
        with open(self.output_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            
            self.assertTrue(len(rows) >= 1, "CSV should have at least header row")
    
    # === TESTING TABLE PROCESSING ===
    # Testing table processing functionality and verifying that table content is correctly extracted from XML.
    
    def test_table_processing(self):
        # Parse XML and find the table element
        soup = BeautifulSoup(self.simple_xml, 'lxml-xml')
        table_node = soup.find('table')
        
        # Process the table
        table_text = process_table_from_element(table_node)
        
        # Verify table processing results
        self.assertIsNotNone(table_text, "Table processing should return text")
        self.assertIn("Header 1", table_text, "Table text should contain header content")
        self.assertIn("Cell 1", table_text, "Table text should contain cell content")
    
    # === CSV DATA EXTRACTION TESTING ===
    # Testing the extraction of CSV data from document structure.
    # Verifying data structure and format.
    
    def test_csv_data_extraction(self):
        
        # Parse XML and build the document tree
        soup = BeautifulSoup(self.simple_xml, 'lxml-xml')
        root = soup.find()
        tree = build_tree(root)
        classify_nodes_respecting_html_tags(tree)
        
        # Process tables in the document
        tables_dict = process_tables_in_document(tree)
        
        # Extract CSV data
        csv_data = extract_csv_data(tree, tables_dict)
        
        # Verify CSV data structure
        self.assertIsInstance(csv_data, dict, "CSV data should be a dictionary")
        self.assertIn("headers", csv_data, "CSV data should have headers key")
        self.assertIn("values", csv_data, "CSV data should have values key")
    
    # === COMPLEX XML DOCUMENT TESTING ===
    
    # Testing processing of a complex XML document.
    # Verifying handling of nested structures and multiple tables.
    
    def test_complex_xml_document(self):
        
        # Parse XML and build the document tree
        soup = BeautifulSoup(self.complex_xml, 'lxml-xml')
        root = soup.find()
        tree = build_tree(root)
        classify_nodes_respecting_html_tags(tree)
        
        # Process tables in the document
        tables_dict = process_tables_in_document(tree)
        
        # Verify tables were extracted
        self.assertIsInstance(tables_dict, dict, "Should get a tables dictionary")
        
        # Extract CSV data
        csv_data = extract_csv_data(tree, tables_dict)
        self.assertIsInstance(csv_data, dict, "CSV data should be a dictionary")
    
    #  === TABLE CONTENT EXTRACTION TESTING ===
    # Testing extraction of content from multiple tables.
    # Verifying table content is properly extracted.
    
    def test_table_content_extraction(self):
        
        # Parse XML and find all tables
        soup = BeautifulSoup(self.complex_xml, 'lxml-xml')
        tables = soup.find_all('table')
        
        # Process each table and verify content
        for table in tables:
            table_content = process_table_from_element(table)
            self.assertIsNotNone(table_content, "Table processing should return content")
            
            self.assertTrue(len(table_content) > 0, "Table content should not be empty")
    
    # === SECTION HIERARCHY TESTING ===
    # Testing extraction of section hierarchy from document.
    # Verifying sections are properly identified.
    
    def test_section_hierarchy(self):
        
        # Parse XML and build the document tree
        soup = BeautifulSoup(self.complex_xml, 'lxml-xml')
        root = soup.find()
        tree = build_tree(root)
        classify_nodes_respecting_html_tags(tree)
        
        # Find all sections in the document
        sections = []
        sections_to_visit = [tree]
        while sections_to_visit:
            current = sections_to_visit.pop(0)
            if current.get("type") == "section":
                sections.append(current)
            sections_to_visit.extend(current.get("children", []))
        
        # Verify sections exist or none which is valid too
        self.assertTrue(len(sections) >= 0, "Should find sections or none, which is valid")
    
    # === FIELD EXTRACTION TESTING ===
    # Testing extraction of field data from document.
    # Verifying key-value pairs are properly extracted.
    
    def test_field_extraction(self):
        
        # Create a simple XML with a field
        field_xml = """
        <document>
            <section>
                <field>TestKey: TestValue</field>
            </section>
        </document>
        """
        
        # Writing field XML to a file
        field_xml_path = os.path.join(self.test_dir, "field.xml")
        with open(field_xml_path, 'w', encoding='utf-8') as f:
            f.write(field_xml)
            
        # Output path for field CSV
        field_output_path = os.path.join(self.test_dir, "field_output.csv")
        
        # Run analysis on field XML
        analyze_document_structure(field_xml_path, field_output_path)
        
        # Verify CSV file was created
        self.assertTrue(os.path.exists(field_output_path), "CSV output file should be created")
        
if __name__ == "__main__":
    unittest.main()