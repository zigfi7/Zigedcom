# Zigedcom
zigedcom - A Modern Genealogy CLI Tool

A flexible, command-line tool for managing, importing, and exporting genealogical data with intelligent merging capabilities.
About The Project

zigedcom is a powerful, Python-based CLI tool designed for genealogists and developers who need a flexible and scriptable way to manage family tree data. It addresses the limitations of traditional software by providing a robust database backend, support for multiple data formats, and smart features to prevent data duplication.

Whether you're migrating from another service, consolidating various data sources, or building your own genealogy-related applications, zigedcom offers the tools you need.
Key Features

    Multiple Backends: Works seamlessly with SQLite and PostgreSQL thanks to SQLAlchemy.

    Versatile Format Support: Import and export data using:

        Standard GEDCOM for universal compatibility.

        Simple CSV for easy, tabular data entry.

        The native ZGE (JSON) format, which supports all features including media.

    Intelligent Data Merging: Avoids duplicates by using fuzzy matching to find and merge records for the same individuals from different sources. Your database remains the single source of truth.

    Flexible Data Model: Uses an event-based structure (birth, death, marriage, occupation, etc.) instead of rigid fields, allowing for rich and varied data for each individual.

    Media Support: Attach photos and documents to individuals using the .zge format (with Base64 encoding).

    Scriptable & Extendable: As a CLI tool, it can be easily integrated into automated workflows and scripts.

Getting Started
Prerequisites

    Python 3.x

Installation & Setup

    Clone the repository:

    git clone [https://github.com/zigfi7/Zigedcom.git](https://github.com/zigfi7/Zigedcom.git)
    cd zigedcom

    Make the run script executable:

    chmod +x run.sh

    The run.sh script will automatically create a Python virtual environment and install all necessary dependencies (SQLAlchemy, fuzzywuzzy, etc.) on the first run.

Usage

The tool is operated via the run.sh script, which passes arguments directly to the Python application.

Basic Syntax:

./run.sh --db-url <DATABASE_URL> [COMMAND] [FILE_PATH]

Examples

    Import from a GEDCOM file:

    ./run.sh --db-url sqlite:///my_family_tree.db --import-gedcom path/to/tree.ged

    Import from a CSV file:

    ./run.sh --db-url sqlite:///my_family_tree.db --import-csv path/to/data.csv

    Export your database to the native ZGE format:

    ./run.sh --db-url sqlite:///my_family_tree.db --export-zge my_tree_backup.zge

    Export your database to a GEDCOM file:

    ./run.sh --db-url sqlite:///my_family_tree.db --export-gedcom my_tree_export.ged

    Using a PostgreSQL database:

    ./run.sh --db-url "postgresql://user:password@localhost/genealogy_db" --import-gedcom path/to/tree.ged

License

Distributed under the Apache 2.0 License. See LICENSE file for more information.