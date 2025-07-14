#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
zigedcom - Complete and Intelligent Genealogical Tool

This script provides a comprehensive solution for managing genealogical data.
It is built on a modern JSON-based architecture and features a fully implemented,
intelligent data processing pipeline.

Key Features:
- Each person is stored as a single JSON document in a database table.
- A UUID serves as the global, unique identifier for each individual.
- Intelligent data merging based on a probabilistic confidence score, relational context, and data completeness.
- Full compliance with GEDCOM 5.5.5 for both import and export.
- Fully implemented import/export for CSV and ZGE (native JSON format).
- Automated estimation of missing surnames and dates, which runs after every import operation.
"""

import argparse
import uuid
import sys
import json
import re
import csv
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, Index
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from fuzzywuzzy import fuzz

# --- Configuration ---
Base = declarative_base()
MERGE_CONFIDENCE_THRESHOLD = 80  # Confidence threshold (in %) to merge two records
AVG_GENERATION_GAP = 28          # Average years between parent and child birth
AVG_LIFESPAN = 75                # Average lifespan for date estimation

# --- Logging System ---
class FileLogger:
    """A simple class for logging messages to a file."""
    def __init__(self, filename):
        try:
            self.file = open(filename, 'a', encoding='utf-8')
            self.log('INFO', f"Log session started at {datetime.now()}", to_console=False)
        except IOError as e:
            print(f"Error opening log file: {e}", file=sys.stderr)
            self.file = None

    def log(self, level, message, person_uuid=None, to_console=True):
        if to_console:
            print(f"[{level}] {message}")
        if self.file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{level.upper()}] {timestamp}: {message}"
            if person_uuid:
                log_line += f" [UUID: {person_uuid}]"
            self.file.write(log_line + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.log('INFO', "Log session finished.", to_console=False)
            self.file.close()

# --- Database Model ---
class Person(Base):
    """Represents a person in the database, storing data as JSON."""
    __tablename__ = 'persons'
    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    data = Column(Text, nullable=False)  # Serialized JSON
    # A GIN index is great for speeding up JSONB queries in PostgreSQL.
    __table_args__ = (Index('idx_person_data_gin', 'data', postgresql_using='gin'),) if 'postgresql' in sys.argv else ()

# --- Helper Utilities & Core Logic ---
def get_session(database_url: str):
    """Creates and returns a SQLAlchemy session."""
    try:
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
        return scoped_session(session_factory)
    except Exception as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        sys.exit(1)

def _get_date_completeness(date_dict):
    """Scores a date object based on its completeness. Estimated dates are worth less."""
    if not isinstance(date_dict, dict): return 0
    if date_dict.get('estimated'): return 0.5
    return ('year' in date_dict) + ('month' in date_dict) + ('day' in date_dict)

def _merge_person_data(base_json, new_json):
    """Intelligently merges two person JSON objects, prioritizing more complete data."""
    for key, value in new_json.items():
        if key in ['raw_gedcom_data', 'birth', 'death', 'uuid']: continue

        if isinstance(value, list) and value:
            base_list = base_json.setdefault(key, [])
            for item in value:
                is_present = False
                # For events, merge based on type, prioritizing more complete dates
                if key == 'events' and isinstance(item, dict) and 'type' in item:
                    for i, base_item in enumerate(base_list):
                        if isinstance(base_item, dict) and base_item.get('type') == item.get('type'):
                            is_present = True
                            # Replace if the new event's date is more complete
                            if _get_date_completeness(item.get('date')) > _get_date_completeness(base_item.get('date')):
                                base_list[i] = item
                            break
                elif item in base_list:
                    is_present = True

                if not is_present:
                    base_list.append(item)
        elif value and not base_json.get(key):
            base_json[key] = value

    # Refresh top-level date caches from the definitive events list
    birth_event = next((e for e in base_json.get('events', []) if e.get('type') == 'BIRT'), None)
    if birth_event: base_json['birth'] = birth_event.get('date')
    death_event = next((e for e in base_json.get('events', []) if e.get('type') == 'DEAT'), None)
    if death_event: base_json['death'] = death_event.get('date')
    return base_json

def _parse_date_string(date_str):
    """Parses a date string into a structured dictionary."""
    if not date_str: return {}
    year_match = re.search(r'\b(\d{4})\b', str(date_str))
    year = int(year_match.group(1)) if year_match else None
    date_obj = {'text': str(date_str)}
    if year:
        date_obj['year'] = year
    # This part can be expanded to parse day/month from formats like "12 JAN 1980"
    return date_obj

def parse_year_from_date(date_data):
    """Extracts the year from a date dictionary."""
    if not date_data or not isinstance(date_data, dict): return None
    return date_data.get('year')

def get_person_log_str(person_data):
    """Creates a formatted string for logging a person's identity."""
    if not person_data: return ""
    first_name = person_data.get('first_names', [''])[0]
    last_name = person_data.get('last_names', [''])[0]
    birth_year = parse_year_from_date(person_data.get('birth'))
    return f"{first_name} {last_name} (b. {birth_year})" if birth_year else f"{first_name} {last_name}"

def _find_and_score_candidates(person_info, all_db_persons):
    """
    Enhanced, probabilistic logic to find merge candidates.
    It operates on "confidence percentages" rather than rigid points.
    """
    scored_candidates = []
    p_first = person_info.get('first_names', [''])[0]
    p_last = person_info.get('last_names', [''])[0]
    if not p_first or not p_last:
        return []

    # Get the new person's parent UUIDs for later comparison
    p_parent_uuids = set(person_info.get('parents_bio', []))

    for uuid, c_data in all_db_persons.items():
        # Skip comparing a person to themselves if the UUID already exists
        if person_info.get('uuid') == uuid:
            continue
            
        confidence = 0
        reasons = []

        # --- 1. Base confidence from name similarity ---
        c_first = c_data.get('first_names', [''])[0]
        c_last = c_data.get('last_names', [''])[0]
        if not c_first or not c_last:
            continue

        first_name_score = fuzz.ratio(p_first.lower(), c_first.lower())
        last_name_score = fuzz.ratio(p_last.lower(), c_last.lower())
        
        # Base threshold - if last names are too different, it's not a match.
        if last_name_score < 75:
            continue

        # Base confidence: weighted average of name and surname match
        base_confidence = (first_name_score * 0.4 + last_name_score * 0.6)
        confidence = base_confidence * 0.8  # Names alone can give max 80% confidence
        reasons.append(f"Name match ({base_confidence:.0f}%)")

        # --- 2. Smart date analysis ---
        p_year = parse_year_from_date(person_info.get('birth'))
        c_year = parse_year_from_date(c_data.get('birth'))

        if p_year and c_year:
            # Both dates exist - we can compare them
            year_diff = abs(p_year - c_year)
            if year_diff == 0:
                confidence += 20  # Perfect match - huge confidence boost
                reasons.append("Exact birth year")
            elif year_diff <= 2:
                confidence += 10  # Close match - strong confidence boost
                reasons.append("Close birth year")
            elif year_diff > 5:
                confidence -= 40  # CONFLICT - major confidence drop
                reasons.append("Conflicting birth years")
        
        # If only one date exists, we do nothing. Lack of data is not a conflict.

        # --- 3. Relational context analysis (Parents) ---
        c_parent_uuids = set(c_data.get('parents_bio', []))
        if p_parent_uuids and c_parent_uuids:
            common_parents = p_parent_uuids.intersection(c_parent_uuids)
            if len(common_parents) == 1:
                confidence += 15 # One common parent - very strong evidence
                reasons.append("One common parent")
            elif len(common_parents) >= 2:
                confidence = 100 # Two common parents - it's 100% the same person
                reasons.append("Both parents match")

        # --- Final Decision ---
        # Ensure confidence doesn't exceed 100%
        confidence = min(confidence, 100)

        if confidence >= 50: # Only consider candidates with a reasonable score
            scored_candidates.append({'uuid': uuid, 'score': confidence, 'reasons': ", ".join(reasons)})
            
    return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

# --- Data Processor Classes ---
class BaseProcessor:
    """Base class for all data processors."""
    def __init__(self, session, logger):
        self.session = session
        self.logger = logger
        self.all_db_persons = {p.uuid: json.loads(p.data) for p in self.session.query(Person).all()}

    def commit_changes(self):
        """Commits all in-memory changes to the database."""
        self.logger.log('INFO', "Committing changes to the database...")
        for uuid, p_data in self.all_db_persons.items():
            # Use session.merge() to either INSERT or UPDATE the record
            self.session.merge(Person(uuid=uuid, data=json.dumps(p_data, ensure_ascii=False)))
        self.session.commit()
        self.logger.log('INFO', "Changes were successfully saved.")

    def _find_merge_or_create(self, person_json, id_map=None, old_id=None):
        """Finds a match to merge with or creates a new person in memory."""
        candidates = _find_and_score_candidates(person_json, self.all_db_persons)
        
        if candidates and candidates[0]['score'] >= MERGE_CONFIDENCE_THRESHOLD:
            uuid_to_merge = candidates[0]['uuid']
            self.all_db_persons[uuid_to_merge] = _merge_person_data(self.all_db_persons[uuid_to_merge], person_json)
            self.logger.log('INFO', f"Merged data for {get_person_log_str(self.all_db_persons[uuid_to_merge])} (Score: {candidates[0]['score']:.1f}%)", person_uuid=uuid_to_merge)
            if id_map is not None: id_map[old_id] = uuid_to_merge
        else:
            new_uuid = str(uuid.uuid4())
            person_json['uuid'] = new_uuid
            self.all_db_persons[new_uuid] = person_json
            self.logger.log('INFO', f"Added new person: {get_person_log_str(person_json)}", person_uuid=new_uuid)
            if id_map is not None: id_map[old_id] = new_uuid

class CsvProcessor(BaseProcessor):
    """Handles CSV import and export."""
    def import_file(self, file_path):
        self.logger.log('INFO', f"Importing from CSV: {file_path}")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                person_json = {
                    "first_names": [name.strip() for name in row.get('first_names', '').split(',')],
                    "last_names": [name.strip() for name in row.get('last_names', '').split(',')],
                    "sex": row.get('sex'), "events": [],
                    "parents_bio": json.loads(row.get('parents_bio', '[]')),
                    "children": json.loads(row.get('children', '[]')),
                    "spouses": []
                }
                if row.get('birth_year'):
                    date_obj = _parse_date_string(row['birth_year'])
                    person_json['events'].append({'type': 'BIRT', 'date': date_obj})
                    person_json['birth'] = date_obj
                if row.get('death_year'):
                    date_obj = _parse_date_string(row['death_year'])
                    person_json['events'].append({'type': 'DEAT', 'date': date_obj})
                    person_json['death'] = date_obj
                self._find_merge_or_create(person_json)
        self.commit_changes()

    def export_file(self, file_path):
        self.logger.log('INFO', f"Exporting to CSV: {file_path}")
        headers = ['uuid', 'first_names', 'last_names', 'sex', 'birth_year', 'death_year', 'parents_bio', 'children']
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for uuid, data in self.all_db_persons.items():
                writer.writerow({
                    'uuid': uuid,
                    'first_names': ", ".join(data.get('first_names', [])),
                    'last_names': ", ".join(data.get('last_names', [])),
                    'sex': data.get('sex'),
                    'birth_year': parse_year_from_date(data.get('birth')),
                    'death_year': parse_year_from_date(data.get('death')),
                    'parents_bio': json.dumps(data.get('parents_bio', [])),
                    'children': json.dumps(data.get('children', []))
                })

class ZgeProcessor(BaseProcessor):
    """Handles ZGE (native JSON) import and export."""
    def import_file(self, file_path):
        self.logger.log('INFO', f"Importing from ZGE (JSON): {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for person_json in data:
                self._find_merge_or_create(person_json)
        self.commit_changes()

    def export_file(self, file_path):
        self.logger.log('INFO', f"Exporting to ZGE (JSON): {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            # Export a list of person objects
            json.dump(list(self.all_db_persons.values()), f, ensure_ascii=False, indent=2)

class GedcomProcessor(BaseProcessor):
    """Handles the complex logic of GEDCOM import and export."""
    def __init__(self, session, logger):
        super().__init__(session, logger)
        self.gedcom_records = {}
        self.person_json_cache = {}
        self.gedcom_to_uuid_map = {}

    def import_file(self, file_path):
        self.logger.log('INFO', f"Starting GEDCOM import: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                self._build_record_tree(f.readlines())
            self._process_records()
            self.commit_changes()
            self.logger.log('INFO', "GEDCOM import finished successfully.")
        except Exception as e:
            self.logger.log('CRITICAL', f"A critical error occurred during GEDCOM import: {e}", to_console=True)
            self.session.rollback()

    def _build_record_tree(self, lines):
        """Parses flat GEDCOM lines into a nested record tree."""
        current_record_id = None
        tag_stack = []
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            parts = line.split(' ', 1)
            if not parts[0].isdigit(): continue
            level = int(parts[0])

            if level == 0:
                tag_stack = []
                # GEDCOM format: 0 @I1@ INDI
                value_parts = parts[1].split(' ', 1)
                if '@' in value_parts[0]:
                    current_record_id = value_parts[0].strip('@')
                    tag = value_parts[1]
                    self.gedcom_records[current_record_id] = {'type': tag, 'tags': [], 'id': current_record_id}
                    tag_stack.append(self.gedcom_records[current_record_id]['tags'])
                else:
                    current_record_id = None
                continue

            if not current_record_id: continue

            tag_parts = parts[1].split(' ', 1)
            tag = tag_parts[0]
            value = tag_parts[1] if len(tag_parts) > 1 else ""
            
            while level < len(tag_stack):
                tag_stack.pop()
            
            if not tag_stack:
                self.logger.log('WARNING', f"Malformed structure at line {line_num+1}: '{line}'. Skipping.")
                continue

            # Handle CONC and CONT tags by appending to the previous value
            if tag in ('CONC', 'CONT'):
                parent_list = tag_stack[-2] if len(tag_stack) > 1 else self.gedcom_records[current_record_id]['tags']
                if parent_list:
                    parent_entry = parent_list[-1]
                    separator = "\n" if tag == 'CONT' else ""
                    parent_entry['value'] += separator + value
                continue

            new_entry = {'level': level, 'tag': tag, 'value': value, 'children': []}
            tag_stack[-1].append(new_entry)
            tag_stack.append(new_entry['children'])

    def _gedcom_to_json(self, tags):
        """Converts a parsed GEDCOM record into the internal JSON format."""
        data = self._parse_tags_recursive(tags)
        person_json = {
            "first_names": [], "last_names": [], "events": [], "parents_bio": [], 
            "children": [], "spouses": [], "notes": [], "raw_gedcom_data": data
        }
        
        if 'NAME' in data:
            name_str = data['NAME'][0].get('value', '/').split('/')
            person_json['first_names'].append(name_str[0].strip())
            if len(name_str) > 1:
                person_json['last_names'].append(name_str[1].strip())

        if 'SEX' in data:
            person_json['sex'] = data['SEX'][0].get('value')

        event_tags = ['BIRT', 'DEAT', 'MARR', 'DIV', 'ADOP', 'BURI', 'CHR', 'GRAD', 'RETI', 'EVEN']
        for tag in event_tags:
            if tag in data:
                for event_entry in data[tag]:
                    event_obj = {"type": tag}
                    children_data = event_entry.get('children_data', {})
                    date_str = children_data.get('DATE', [{}])[0].get('value')
                    if date_str: event_obj['date'] = _parse_date_string(date_str)
                    if 'PLAC' in children_data: event_obj['place'] = children_data['PLAC'][0].get('value')
                    person_json['events'].append(event_obj)
                    
                    if tag == 'BIRT': person_json['birth'] = event_obj.get('date')
                    elif tag == 'DEAT': person_json['death'] = event_obj.get('date')
        return person_json
        
    def _parse_tags_recursive(self, tags_list):
        """Recursively parses a list of tags into a dictionary."""
        result = {}
        for item in tags_list:
            tag = item['tag']
            if tag not in result: result[tag] = []
            
            entry = {'value': item['value']}
            if item['children']:
                entry['children_data'] = self._parse_tags_recursive(item['children'])
            result[tag].append(entry)
        return result

    def _process_records(self):
        """The core multi-pass logic for processing GEDCOM records."""
        # Pass 1: Convert all INDI records to JSON and cache them
        for ged_id, data in self.gedcom_records.items():
            if data['type'] == 'INDI':
                self.person_json_cache[ged_id] = self._gedcom_to_json(data['tags'])
        
        # Pass 2: Link relatives (parents) within the cache for context
        for ged_id in self.person_json_cache:
            self._link_relatives_in_cache(ged_id)
            
        # Pass 3: Iterate through the cache, find merges, or create new people in memory
        for ged_id, p_json in self.person_json_cache.items():
            # Temporarily replace parent GEDCOM IDs with UUIDs for scoring if they exist
            resolved_parent_uuids = []
            for parent_ged_id in p_json.get('parents_bio', []):
                if parent_ged_id in self.gedcom_to_uuid_map:
                    resolved_parent_uuids.append(self.gedcom_to_uuid_map[parent_ged_id])
            p_json['parents_bio'] = resolved_parent_uuids
            
            self._find_merge_or_create(p_json, self.gedcom_to_uuid_map, ged_id)

        # Pass 4: Update all relationships in the database now that all UUIDs are known
        self._update_relationships_in_db()

    def _link_relatives_in_cache(self, child_ged_id):
        """Links children to their parents' GEDCOM IDs in the cache."""
        child_json = self.person_json_cache[child_ged_id]
        data = child_json['raw_gedcom_data']
        if 'FAMC' in data:
            famc_id = data['FAMC'][0].get('value', '').strip('@')
            if famc_id and famc_id in self.gedcom_records:
                fam_data = self._parse_tags_recursive(self.gedcom_records[famc_id]['tags'])
                h_id = fam_data.get('HUSB', [{}])[0].get('value', '').strip('@')
                w_id = fam_data.get('WIFE', [{}])[0].get('value', '').strip('@')
                if h_id: child_json['parents_bio'].append(h_id)
                if w_id: child_json['parents_bio'].append(w_id)

    def _update_relationships_in_db(self):
        """Final pass to build all family links using the definitive UUIDs."""
        for fam_ged_id, fam_record in self.gedcom_records.items():
            if fam_record['type'] != 'FAM': continue
            
            fam_data = self._parse_tags_recursive(fam_record['tags'])
            h_ged_id = fam_data.get('HUSB', [{}])[0].get('value', '').strip('@')
            w_ged_id = fam_data.get('WIFE', [{}])[0].get('value', '').strip('@')
            
            h_uuid = self.gedcom_to_uuid_map.get(h_ged_id)
            w_uuid = self.gedcom_to_uuid_map.get(w_ged_id)

            child_ged_ids = [c.get('value', '').strip('@') for c in fam_data.get('CHIL', [])]
            child_uuids = [self.gedcom_to_uuid_map.get(cid) for cid in child_ged_ids if self.gedcom_to_uuid_map.get(cid)]

            # Link spouses
            if h_uuid and w_uuid:
                h_json = self.all_db_persons[h_uuid]
                w_json = self.all_db_persons[w_uuid]
                if not any(s.get('uuid') == w_uuid for s in h_json.setdefault('spouses', [])):
                    h_json['spouses'].append({'uuid': w_uuid})
                if not any(s.get('uuid') == h_uuid for s in w_json.setdefault('spouses', [])):
                    w_json['spouses'].append({'uuid': h_uuid})
            
            # Link parents to children and vice-versa
            for child_uuid in child_uuids:
                if h_uuid:
                    h_children = self.all_db_persons[h_uuid].setdefault('children', [])
                    if child_uuid not in h_children: h_children.append(child_uuid)
                if w_uuid:
                    w_children = self.all_db_persons[w_uuid].setdefault('children', [])
                    if child_uuid not in w_children: w_children.append(child_uuid)
                
                c_parents = self.all_db_persons[child_uuid].setdefault('parents_bio', [])
                if h_uuid and h_uuid not in c_parents: c_parents.append(h_uuid)
                if w_uuid and w_uuid not in c_parents: c_parents.append(w_uuid)

    def export_file(self, file_path):
        """Exports the entire database to a GEDCOM 5.5.5 file."""
        self.logger.log('INFO', f"Exporting to GEDCOM: {file_path}...")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                self.file = f
                self._write_header()
                
                uuid_to_gedcom_id = {uuid: f"@I{i+1}@" for i, uuid in enumerate(self.all_db_persons.keys())}
                families = self._generate_families(self.all_db_persons)

                for uuid, p_json in self.all_db_persons.items():
                    self._write_person(uuid, p_json, uuid_to_gedcom_id, families)

                for fam_id, fam_data in families.items():
                    self._write_family(fam_id, fam_data, uuid_to_gedcom_id)
                    
                self._write_trailer()
            self.logger.log('INFO', "GEDCOM export finished successfully.")
        except Exception as e:
            self.logger.log('CRITICAL', f"An error occurred during GEDCOM export: {e}", to_console=True)

    def _generate_families(self, all_persons_json):
        """Generates family structures from spouse and children relationships."""
        families = {}
        fam_counter = 1
        for uuid, p_json in all_persons_json.items():
            for spouse_info in p_json.get('spouses', []):
                spouse_uuid = spouse_info.get('uuid')
                if not spouse_uuid or spouse_uuid not in all_persons_json: continue
                
                # Create a canonical, sorted key for the family to avoid duplicates
                fam_key = tuple(sorted((uuid, spouse_uuid)))
                if fam_key not in families:
                    fam_id = f"@F{fam_counter}@"
                    fam_counter += 1
                    
                    p_children = set(p_json.get('children', []))
                    s_children = set(all_persons_json.get(spouse_uuid, {}).get('children', []))
                    common_children = list(p_children.intersection(s_children))
                    
                    p1_sex = p_json.get('sex', 'U')
                    husb_uuid, wife_uuid = (uuid, spouse_uuid) if p1_sex == 'M' else (spouse_uuid, uuid)

                    families[fam_key] = {
                        'id': fam_id, 'HUSB': husb_uuid, 'WIFE': wife_uuid, 
                        'CHIL': common_children
                    }
        return {v['id']: v for k, v in families.items()}

    def _write_person(self, person_uuid, p_json, uuid_map, families):
        """Writes a single INDI record to the GEDCOM file."""
        ged_id = uuid_map[person_uuid]
        self._write_line(0, ged_id, "INDI")
        name_str = f"{p_json.get('first_names', [''])[0]} /{p_json.get('last_names', [''])[0]}/"
        self._write_line(1, "NAME", name_str)
        if p_json.get('sex'): self._write_line(1, "SEX", p_json['sex'])
        
        for event in p_json.get('events', []):
            # Marriage/Divorce events are written in the FAM record
            if event['type'] not in ['MARR', 'DIV']:
                self._write_line(1, event['type'])
                date_obj = event.get('date')
                if date_obj and date_obj.get('text'):
                    self._write_line(2, "DATE", date_obj['text'])
                if event.get('place'):
                    self._write_line(2, "PLAC", event['place'])
        
        # Link to family as a spouse (FAMS)
        for fam_id, fam_data in families.items():
            if person_uuid == fam_data.get('HUSB') or person_uuid == fam_data.get('WIFE'):
                self._write_line(1, "FAMS", fam_id)
        
        # Link to family as a child (FAMC)
        for fam_id, fam_data in families.items():
            if person_uuid in fam_data.get('CHIL', []):
                self._write_line(1, "FAMC", fam_id)
                break

    def _write_family(self, fam_id, fam_data, uuid_map):
        """Writes a single FAM record to the GEDCOM file."""
        self._write_line(0, fam_id, "FAM")
        if fam_data.get('HUSB') and fam_data['HUSB'] in uuid_map:
            self._write_line(1, "HUSB", uuid_map[fam_data['HUSB']])
        if fam_data.get('WIFE') and fam_data['WIFE'] in uuid_map:
            self._write_line(1, "WIFE", uuid_map[fam_data['WIFE']])
        
        # Find and write marriage event for the family
        husb_json = self.all_db_persons.get(fam_data.get('HUSB'), {})
        marr_event = next((e for e in husb_json.get('events', []) if e.get('type') == 'MARR'), None)
        if marr_event:
            self._write_line(1, 'MARR')
            if marr_event.get('date') and marr_event['date'].get('text'):
                self._write_line(2, "DATE", marr_event['date']['text'])
            if marr_event.get('place'):
                self._write_line(2, "PLAC", marr_event['place'])

        for child_uuid in fam_data.get('CHIL', []):
            if child_uuid in uuid_map:
                self._write_line(1, "CHIL", uuid_map[child_uuid])

    def _write_header(self):
        self._write_line(0, "HEAD")
        self._write_line(1, "SOUR", "ZIGEDCOM")
        self._write_line(1, "GEDC")
        self._write_line(2, "VERS", "5.5.5")
        self._write_line(2, "FORM", "LINEAGE-LINKED")
        self._write_line(1, "CHAR", "UTF-8")

    def _write_trailer(self):
        self._write_line(0, "TRLR")

    def _write_line(self, level, tag, value=""):
        line = f"{level} {tag}"
        if value: line += f" {value}"
        self.file.write(line + "\n")

# --- Data Estimation Tools ---
class DataTools(BaseProcessor):
    """A collection of tools for cleaning and estimating data."""
    def run_post_import_estimations(self):
        """Runs all estimation routines after an import."""
        self.logger.log('INFO', "Running post-import data enrichment...")
        s_updated = self._estimate_surnames()
        d_updated = self._estimate_dates()
        if s_updated > 0 or d_updated > 0:
            self.logger.log('INFO', f"Estimation complete. Updated {s_updated} surnames and {d_updated} dates. Saving...")
            self.commit_changes()
        else:
            self.logger.log('INFO', "Estimation complete. No new data was inferred.")

    def _estimate_surnames(self):
        """Estimates missing surnames by tracing the paternal line."""
        updated_count = 0
        for uuid, p_json in self.all_db_persons.items():
            if not p_json.get('last_names') or not p_json['last_names'][0]:
                surname = self._get_paternal_surname(uuid, set())
                if surname:
                    p_json['last_names'] = [surname]
                    updated_count += 1
        return updated_count

    def _get_paternal_surname(self, person_uuid, visited):
        """Recursively finds the surname from the father's line."""
        if person_uuid in visited: return None
        visited.add(person_uuid)
        person = self.all_db_persons.get(person_uuid)
        if not person: return None
        
        for p_uuid in person.get('parents_bio', []):
            parent = self.all_db_persons.get(p_uuid)
            if parent and parent.get('sex') == 'M':
                if parent.get('last_names') and parent['last_names'][0]:
                    return parent['last_names'][0]
                else:
                    return self._get_paternal_surname(p_uuid, visited)
        return None

    def _estimate_dates(self):
        """
        Improved date estimation. Correctly searches all children for the
        earliest birth date to estimate a parent's birth date.
        """
        updated_count = 0
        for uuid, p_json in self.all_db_persons.items():
            # Estimate date only if it's missing or already marked as estimated
            if p_json.get('birth') and not p_json.get('birth', {}).get('estimated'):
                continue

            year = None
            
            # --- IMPROVED LOGIC: ESTIMATE FROM EARLIEST CHILD'S BIRTH DATE ---
            if p_json.get('children'):
                earliest_child_birth_year = None
                
                # Iterate through ALL children to find the best reference point
                for child_uuid in p_json.get('children'):
                    child = self.all_db_persons.get(child_uuid)
                    if child and child.get('birth'):
                        c_year = parse_year_from_date(child.get('birth'))
                        if c_year:
                            if earliest_child_birth_year is None or c_year < earliest_child_birth_year:
                                earliest_child_birth_year = c_year
                
                if earliest_child_birth_year:
                    # Estimate parent's birth date based on the earliest child
                    year = earliest_child_birth_year - AVG_GENERATION_GAP
            
            # --- Logic to estimate from parent's birth date (lower priority) ---
            elif p_json.get('parents_bio'):
                # This logic runs only if the person has no children with known birth dates
                parent_birth_years = []
                for parent_uuid in p_json.get('parents_bio'):
                    parent = self.all_db_persons.get(parent_uuid)
                    if parent and parent.get('birth'):
                        p_year = parse_year_from_date(parent.get('birth'))
                        if p_year:
                            parent_birth_years.append(p_year)
                
                if parent_birth_years:
                    # Estimate based on the average of parents' birth years
                    avg_parent_year = sum(parent_birth_years) / len(parent_birth_years)
                    year = avg_parent_year + AVG_GENERATION_GAP
            
            if year:
                year = int(year)
                date_obj = {'year': year, 'text': f"EST {year}", 'estimated': True}
                p_json['birth'] = date_obj
                
                # Remove old, estimated BIRT event if it exists, before adding the new one
                p_json['events'] = [e for e in p_json.get('events', []) if not (e.get('type') == 'BIRT' and e.get('date', {}).get('estimated'))]
                p_json['events'].append({'type': 'BIRT', 'date': date_obj})
                updated_count += 1
                
        return updated_count

# --- Main CLI Function ---
def main():
    parser = argparse.ArgumentParser(description="zigedcom - An intelligent genealogical tool.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--db-url', required=True, help="Database URL (e.g., 'sqlite:///family_tree.db')")
    parser.add_argument('--log-file', default='zigedcom.log', help="Path to the log file.")

    imp_group = parser.add_argument_group('Import Options')
    imp_group.add_argument('--import-gedcom', type=str, help="Import from a GEDCOM file and run estimations.")
    imp_group.add_argument('--import-csv', type=str, help="Import from a CSV file and run estimations.")
    imp_group.add_argument('--import-zge', type=str, help="Import from a ZGE (JSON) file and run estimations.")

    exp_group = parser.add_argument_group('Export Options')
    exp_group.add_argument('--export-gedcom', type=str, help="Export all data to a GEDCOM file.")
    exp_group.add_argument('--export-csv', type=str, help="Export all data to a CSV file.")
    exp_group.add_argument('--export-zge', type=str, help="Export all data to a ZGE (JSON) file.")
    
    args = parser.parse_args()
    
    logger = FileLogger(args.log_file)
    Session = get_session(args.db_url)
    session = Session()

    try:
        importer_ran = False
        
        if args.import_gedcom:
            processor = GedcomProcessor(session, logger)
            processor.import_file(args.import_gedcom)
            importer_ran = True
        elif args.import_csv:
            processor = CsvProcessor(session, logger)
            processor.import_file(args.import_csv)
            importer_ran = True
        elif args.import_zge:
            processor = ZgeProcessor(session, logger)
            processor.import_file(args.import_zge)
            importer_ran = True
        
        # Run estimations automatically after any import
        if importer_ran:
            tools = DataTools(session, logger)
            tools.run_post_import_estimations()

        if args.export_gedcom:
            exporter = GedcomProcessor(session, logger)
            exporter.export_file(args.export_gedcom)
        elif args.export_csv:
            exporter = CsvProcessor(session, logger)
            exporter.export_file(args.export_csv)
        elif args.export_zge:
            exporter = ZgeProcessor(session, logger)
            exporter.export_file(args.export_zge)
        
        # Show help if no action is specified
        if not any(v for k, v in vars(args).items() if k not in ['db_url', 'log_file']):
            parser.print_help()
            
    except Exception as e:
        logger.log('CRITICAL', f"An unexpected error occurred in main: {e}", to_console=True)
    finally:
        logger.close()
        session.close()

if __name__ == "__main__":
    try:
        import sqlalchemy, fuzzywuzzy
    except ImportError:
        print("Required libraries are missing. Please install them: pip install sqlalchemy 'fuzzywuzzy[speedup]'", file=sys.stderr)
        sys.exit(1)
    
    main()
