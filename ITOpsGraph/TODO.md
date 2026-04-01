# ITOpsGraph - Pending Tasks

## Documentation & Content

### 1. Collect HTML documentation from all ITOps products
- Status: In Progress
- Gather HTML documentation from product websites and portals
- Convert to markdown for ChromaDB indexing

### 2. Improve Proto-based products schema information
- Status: Pending
- Enhance schema details for AWS, Azure, GCP, Infragraph, HCP Packer, HCP Terraform
- Add better field descriptions and type documentation

### 3. Enhance schema mapping for Proto-based products
- Status: Pending
- Create cross-product type mappings for proto-based products
- Currently GraphQL products have better mapping coverage

## Wiki Quality

### 4. Expand ITOps ontology with missing concepts
- Status: Pending
- Add new semantic classes for emerging ITOps concepts
- Review and update existing class hierarchies

### 5. Add glossary terms for under-represented products
- Status: Pending
- Extract glossary terms from AWS, Azure, GCP documentation
- Balance term coverage across all products

### 6. Create GraphRAG solution for wiki-based schema mapping
- Status: Pending
- Implement graph-based retrieval augmented generation
- Use wiki content and Neo4j graph for intelligent schema discovery

### 7. Improve wiki page quality with better descriptions
- Status: Pending
- Add more detailed descriptions to type and field pages
- Improve product overview pages with use cases

### 8. Add field-level documentation to wiki type pages
- Status: Pending
- Include field descriptions from GraphQL/Proto schemas
- Show field relationships and constraints

### 9. Create wiki pages for Glossary terms and Ontology classes
- Status: Pending
- Generate dedicated wiki pages for each glossary term
- Create wiki pages for ontology classes with relationships

### 10. Validate and improve cross-product type mappings
- Status: Pending
- Review existing TYPE_MAPPING relationships for accuracy
- Add missing mappings between semantically related types

---

## Completed

- [x] GraphQL Schema Snippets - Added to wiki type pages
- [x] Related Types from Neo4j - Cross-product mappings shown on pages
- [x] ChromaDB Documentation - Smart CamelCase-aware filtering (60% coverage)
- [x] Main page protection - Skipped by default during wiki regeneration
- [x] Full wiki upload - 1716 pages uploaded successfully
