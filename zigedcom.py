#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
zigedcom v3 - Zaawansowane narzędzie CLI do zarządzania danymi genealogicznymi.

Obsługuje import/eksport dla formatów GEDCOM, CSV i własnego formatu ZIG-JSON.
- Nowa, elastyczna struktura bazy danych oparta na wydarzeniach.
- Obsługa mediów w formacie Base64.
- Inteligentne mapowanie kolumn przy imporcie z CSV.
- Ulepszony parser GEDCOM.
"""

import argparse
import csv
import uuid
import sys
import json
import base64
from sqlalchemy import create_engine, Column, String, ForeignKey, Text, LargeBinary
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, scoped_session
from fuzzywuzzy import fuzz

# --- Konfiguracja ---
Base = declarative_base()
SIMILARITY_THRESHOLD = 85
MAX_MEDIA_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# --- Modele Bazy Danych (v3) ---

class Person(Base):
    """Model reprezentujący osobę w bazie danych."""
    __tablename__ = 'persons'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    sex = Column(String(1), nullable=True)  # M, F, U
    gedcom_id = Column(String, unique=True, nullable=True, index=True)
    events = relationship("Event", back_populates="person", cascade="all, delete-orphan")
    media_files = relationship("Media", back_populates="person", cascade="all, delete-orphan")
    name_variations = relationship("NameVariation", back_populates="person", cascade="all, delete-orphan")
    # ... relacje zdefiniowane niżej ...

class Event(Base):
    """Model dla wydarzeń w życiu osoby (np. narodziny, śmierć, chrzest)."""
    __tablename__ = 'events'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = Column(String, ForeignKey('persons.id'), nullable=False)
    event_type = Column(String, nullable=False)  # BIRT, DEAT, MARR, OCCU, etc.
    event_date = Column(String, nullable=True)
    event_place = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    person = relationship("Person", back_populates="events")

class Media(Base):
    """Model dla plików multimedialnych powiązanych z osobą."""
    __tablename__ = 'media'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = Column(String, ForeignKey('persons.id'), nullable=False)
    title = Column(String, nullable=True)
    file_name = Column(String, nullable=True)
    base64_data = Column(LargeBinary, nullable=False)
    person = relationship("Person", back_populates="media_files")

class NameVariation(Base):
    """Model dla alternatywnych zapisów imion i nazwisk."""
    __tablename__ = 'name_variations'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = Column(String, ForeignKey('persons.id'), nullable=False)
    alternative_name = Column(String)
    alternative_surname = Column(String)
    variation_type = Column(String)
    person = relationship("Person", back_populates="name_variations")

class Relationship(Base):
    """Model reprezentujący relację rodzic-dziecko."""
    __tablename__ = 'relationships'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_id = Column(String, ForeignKey('persons.id'), nullable=False)
    child_id = Column(String, ForeignKey('persons.id'), nullable=False)

class Marriage(Base):
    """Model reprezentujący małżeństwo."""
    __tablename__ = 'marriages'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_a_id = Column(String, ForeignKey('persons.id'), nullable=False)
    person_b_id = Column(String, ForeignKey('persons.id'), nullable=False)
    marriage_date = Column(String, nullable=True)
    marriage_place = Column(String, nullable=True)

# Definicje relacji dla Person (po zdefiniowaniu wszystkich klas)
Person.children_relations = relationship("Relationship", foreign_keys="[Relationship.parent_id]", back_populates="parent", cascade="all, delete-orphan")
Person.parent_relations = relationship("Relationship", foreign_keys="[Relationship.child_id]", back_populates="child", cascade="all, delete-orphan")
Person.marriages1 = relationship("Marriage", foreign_keys="[Marriage.person_a_id]", back_populates="person_a", cascade="all, delete-orphan")
Person.marriages2 = relationship("Marriage", foreign_keys="[Marriage.person_b_id]", back_populates="person_b", cascade="all, delete-orphan")
Relationship.parent = relationship("Person", foreign_keys=[Relationship.parent_id], back_populates="children_relations")
Relationship.child = relationship("Person", foreign_keys=[Relationship.child_id], back_populates="parent_relations")
Marriage.person_a = relationship("Person", foreign_keys=[Marriage.person_a_id], back_populates="marriages1")
Marriage.person_b = relationship("Person", foreign_keys=[Marriage.person_b_id], back_populates="marriages2")


# --- Logika Bazy Danych i Narzędzia Pomocnicze ---

def get_session(database_url: str):
    """Tworzy i zwraca sesję SQLAlchemy."""
    try:
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
        return scoped_session(session_factory)
    except Exception as e:
        print(f"Błąd połączenia z bazą danych: {e}", file=sys.stderr)
        sys.exit(1)

def find_similar_person(session, first_name, last_name):
    # ... (bez zmian)
    candidates = session.query(Person).all()
    for candidate in candidates:
        name_ratio = fuzz.ratio(candidate.first_name.lower(), first_name.lower())
        surname_ratio = fuzz.ratio(candidate.last_name.lower(), last_name.lower())
        if name_ratio > SIMILARITY_THRESHOLD and surname_ratio > SIMILARITY_THRESHOLD:
            return candidate
        for variation in candidate.name_variations:
            var_name_ratio = fuzz.ratio(variation.alternative_name.lower(), first_name.lower())
            var_surname_ratio = fuzz.ratio(variation.alternative_surname.lower(), last_name.lower())
            if var_name_ratio > SIMILARITY_THRESHOLD and var_surname_ratio > SIMILARITY_THRESHOLD:
                return candidate
    return None

# --- Inteligentny Import CSV ---

def get_csv_column_map(header):
    """Tworzy mapowanie nagłówków CSV na pola w bazie danych."""
    HEADER_MAPPINGS = {
        'first_name': ['first name', 'imię', 'given', 'imie'],
        'last_name': ['last name', 'nazwisko', 'surname'],
        'birth_date': ['birth date', 'data urodzenia', 'urodziny', 'birth'],
        'birth_place': ['birth place', 'miejsce urodzenia'],
        'death_date': ['death date', 'data śmierci', 'śmierć', 'death'],
        'death_place': ['death place', 'miejsce śmierci'],
        'sex': ['sex', 'płeć', 'plec'],
        'notes': ['notes', 'notatki', 'uwagi', 'description', 'opis']
    }
    column_map = {}
    for col_name in header:
        for db_field, variations in HEADER_MAPPINGS.items():
            if col_name.lower().strip() in variations:
                column_map[col_name] = db_field
                break
    return column_map

def import_csv_to_db(session, csv_file: str):
    """Inteligentnie importuje dane z pliku CSV do bazy danych."""
    print(f"Rozpoczynam inteligentny import z pliku CSV: {csv_file}...")
    try:
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            column_map = get_csv_column_map(header)
            
            if 'first_name' not in column_map.values() or 'last_name' not in column_map.values():
                print("Błąd: Plik CSV musi zawierać kolumny z imieniem i nazwiskiem.", file=sys.stderr)
                return

            dict_reader = csv.DictReader(f, fieldnames=header)
            for row in dict_reader:
                mapped_row = {db_field: row[csv_col] for csv_col, db_field in column_map.items()}
                first_name = mapped_row.get('first_name', '').strip()
                last_name = mapped_row.get('last_name', '').strip()

                if not first_name or not last_name: continue

                person = find_similar_person(session, first_name, last_name)
                if not person:
                    person = Person(first_name=first_name, last_name=last_name, sex=mapped_row.get('sex'))
                    session.add(person)
                    print(f"  Dodaję nową osobę: {first_name} {last_name}")
                
                # Dodawanie wydarzeń zmapowanych z CSV
                if mapped_row.get('birth_date') or mapped_row.get('birth_place'):
                    session.add(Event(person=person, event_type='BIRT', event_date=mapped_row.get('birth_date'), event_place=mapped_row.get('birth_place')))
                if mapped_row.get('death_date') or mapped_row.get('death_place'):
                    session.add(Event(person=person, event_type='DEAT', event_date=mapped_row.get('death_date'), event_place=mapped_row.get('death_place')))
                if mapped_row.get('notes'):
                    session.add(Event(person=person, event_type='NOTE', description=mapped_row.get('notes')))

        session.commit()
        print("Import z CSV zakończony pomyślnie.")
    except Exception as e:
        print(f"Wystąpił błąd podczas importu CSV: {e}", file=sys.stderr)
        session.rollback()


# --- Funkcje Importu/Eksportu GEDCOM (Przebudowane) ---

def import_gedcom_to_db(session, gedcom_file: str):
    """Importuje dane z pliku GEDCOM do bazy danych (ulepszony parser)."""
    print(f"Rozpoczynam import z pliku GEDCOM (v3): {gedcom_file}...")
    try:
        with open(gedcom_file, 'r', encoding='utf-8-sig') as f: # utf-8-sig do obsługi BOM
            lines = f.readlines()

        # Krok 1: Parsowanie do struktury pośredniej
        records = {}
        current_record = None
        for line in lines:
            line = line.strip()
            if not line: continue

            parts = line.split(' ', 1)
            if not parts[0].isdigit(): continue

            level = int(parts[0])
            
            if level == 0:
                if len(parts) < 2: continue
                
                tag_and_value = parts[1]
                
                if '@' in tag_and_value:
                    value_parts = tag_and_value.split(' ', 1)
                    record_id = value_parts[0].strip('@')
                    
                    if len(value_parts) > 1:
                        record_type = value_parts[1]
                        if record_type in ('INDI', 'FAM'):
                            current_record = {'type': record_type, 'id': record_id, 'tags': []}
                            records[record_id] = current_record
                        else:
                            current_record = None
                    else:
                        current_record = None
                else:
                    current_record = None
                continue
            
            if current_record:
                if len(parts) < 2: continue
                tag_and_value = parts[1]
                tag_parts = tag_and_value.split(' ', 1)
                tag = tag_parts[0]
                value = tag_parts[1] if len(tag_parts) > 1 else ""
                current_record['tags'].append({'level': level, 'tag': tag, 'value': value})

        # Krok 2: Przetwarzanie osób i rodzin
        gedcom_to_db_id_map = {}
        # Pass 1: Create or find Person objects
        for ged_id, data in records.items():
            if data['type'] != 'INDI': continue
            
            person_data = {'gedcom_id': ged_id}
            last_tag = None
            for tag_data in data['tags']:
                if tag_data['level'] == 1:
                    last_tag = tag_data['tag']
                    if tag_data['tag'] == 'NAME':
                        names = tag_data['value'].split('/')
                        person_data['first_name'] = names[0].strip()
                        person_data['last_name'] = names[1].strip() if len(names) > 1 else ""
                    elif tag_data['tag'] == 'SEX':
                        person_data['sex'] = tag_data['value']
                elif tag_data['level'] == 2 and last_tag is not None:
                    event_type = last_tag
                    if event_type not in person_data: person_data[event_type] = {}
                    person_data[event_type][tag_data['tag']] = tag_data['value']
            
            first_name = person_data.get('first_name', 'Nieznane')
            last_name = person_data.get('last_name', 'Nieznane')

            # --- ZMODYFIKOWANA LOGIKA ---
            # 1. Spróbuj znaleźć podobną osobę w naszej bazie
            person = find_similar_person(session, first_name, last_name)
            
            if person:
                # Znaleziono podobną osobę, użyj jej
                print(f"  Znaleziono podobną osobę dla GEDCOM ID {ged_id}: {person.first_name} {person.last_name}. Scalam dane.")
                # Jeśli nasza osoba nie ma jeszcze ID z GEDCOM, przypisz je.
                if not person.gedcom_id:
                    person.gedcom_id = ged_id
            else:
                # 2. Jeśli nie ma podobnej, sprawdź czy ID z GEDCOM już istnieje
                person = session.query(Person).filter_by(gedcom_id=ged_id).first()
                if not person:
                    # 3. Dopiero teraz stwórz nową osobę
                    print(f"  Nie znaleziono podobnej osoby. Dodaję nową z GEDCOM ID {ged_id}: {first_name} {last_name}")
                    person = Person(
                        first_name=first_name,
                        last_name=last_name,
                        sex=person_data.get('sex'),
                        gedcom_id=ged_id
                    )
                    session.add(person)
            
            session.flush()
            gedcom_to_db_id_map[ged_id] = person.id
            
            # Dodaj/zaktualizuj wydarzenia dla znalezionej lub nowej osoby
            for event_type, event_data in person_data.items():
                if isinstance(event_data, dict):
                    # Sprawdź czy podobne wydarzenie już nie istnieje, aby uniknąć duplikatów
                    existing_event = session.query(Event).filter_by(
                        person_id=person.id,
                        event_type=event_type,
                        event_date=event_data.get('DATE')
                    ).first()
                    if not existing_event:
                        session.add(Event(
                            person=person,
                            event_type=event_type,
                            event_date=event_data.get('DATE'),
                            event_place=event_data.get('PLAC'),
                            description=event_data.get('NOTE')
                        ))
        
        # Pass 2: Create relationships
        for ged_id, data in records.items():
            if data['type'] != 'FAM': continue
            
            family_data = {}
            for tag_data in data['tags']:
                if tag_data['tag'] in ('HUSB', 'WIFE', 'CHIL'):
                    if tag_data['tag'] not in family_data: family_data[tag_data['tag']] = []
                    family_data[tag_data['tag']].append(tag_data['value'].strip('@'))
            
            h_id = gedcom_to_db_id_map.get(family_data.get('HUSB', [None])[0])
            w_id = gedcom_to_db_id_map.get(family_data.get('WIFE', [None])[0])
            
            if h_id and w_id:
                if not session.query(Marriage).filter(((Marriage.person_a_id == h_id) & (Marriage.person_b_id == w_id)) | ((Marriage.person_a_id == w_id) & (Marriage.person_b_id == h_id))).first():
                    session.add(Marriage(person_a_id=h_id, person_b_id=w_id))
            
            for child_ged_id in family_data.get('CHIL', []):
                c_id = gedcom_to_db_id_map.get(child_ged_id)
                if not c_id: continue
                if h_id and not session.query(Relationship).filter_by(parent_id=h_id, child_id=c_id).first():
                    session.add(Relationship(parent_id=h_id, child_id=c_id))
                if w_id and not session.query(Relationship).filter_by(parent_id=w_id, child_id=c_id).first():
                    session.add(Relationship(parent_id=w_id, child_id=c_id))

        session.commit()
        print("Import z GEDCOM v3 zakończony pomyślnie.")
    except Exception as e:
        print(f"Wystąpił błąd podczas importu GEDCOM: {e}", file=sys.stderr)
        session.rollback()

def export_db_to_gedcom(session, gedcom_file: str):
    """Eksportuje dane z bazy do pliku GEDCOM (ulepszona wersja)."""
    # ... (logika podobna do poprzedniej, ale odczytująca z nowej struktury Event)
    print("Eksport do GEDCOM nie został jeszcze w pełni zrefaktoryzowany dla v3.")


# --- Funkcje Importu/Eksportu ZIG-JSON ---

def export_db_to_zig_json(session, json_file: str):
    """Eksportuje bazę danych do własnego formatu ZIG-JSON."""
    print(f"Eksportuję do formatu ZIG-JSON: {json_file}...")
    output = {}
    persons = session.query(Person).all()

    for p in persons:
        person_data = {
            "name": {"first": p.first_name, "last": p.last_name},
            "sex": p.sex,
            "events": [
                {"type": e.event_type, "date": e.event_date, "place": e.event_place, "desc": e.description}
                for e in p.events
            ],
            "media": [
                {"title": m.title, "filename": m.file_name, "data_b64": base64.b64encode(m.base64_data).decode('utf-8')}
                for m in p.media_files
            ],
            "variations": [
                {"type": v.variation_type, "first": v.alternative_name, "last": v.alternative_surname}
                for v in p.name_variations
            ],
            "relatives": {
                "spouses": [m.person_b_id if m.person_a_id == p.id else m.person_a_id for m in (p.marriages1 + p.marriages2)],
                "children": [r.child_id for r in p.children_relations],
                "parents": [r.parent_id for r in p.parent_relations]
            }
        }
        output[p.id] = person_data
    
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("Eksport do ZIG-JSON zakończony pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas eksportu do ZIG-JSON: {e}", file=sys.stderr)

def import_zig_json_to_db(session, json_file: str):
    """Importuje dane z pliku ZIG-JSON do bazy danych."""
    print(f"Importuję z formatu ZIG-JSON: {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Pass 1: Create persons, events, media
        for person_id, p_data in data.items():
            person = Person(
                id=person_id,
                first_name=p_data['name']['first'],
                last_name=p_data['name']['last'],
                sex=p_data.get('sex')
            )
            session.add(person)

            for event in p_data.get('events', []):
                session.add(Event(person_id=person_id, **event))
            
            for media in p_data.get('media', []):
                b64_data = media.get('data_b64', '').encode('utf-8')
                if len(b64_data) > MAX_MEDIA_SIZE_BYTES:
                    print(f"Ostrzeżenie: Plik media '{media.get('title')}' dla osoby {person_id} jest za duży ({len(b64_data)} > {MAX_MEDIA_SIZE_BYTES}) i został pominięty.", file=sys.stderr)
                    continue
                session.add(Media(person_id=person_id, title=media.get('title'), file_name=media.get('filename'), base64_data=base64.b64decode(b64_data)))

        # Pass 2: Create relationships
        for person_id, p_data in data.items():
            relatives = p_data.get('relatives', {})
            for spouse_id in relatives.get('spouses', []):
                # Avoid duplicates
                if not session.query(Marriage).filter(((Marriage.person_a_id == person_id) & (Marriage.person_b_id == spouse_id)) | ((Marriage.person_a_id == spouse_id) & (Marriage.person_b_id == person_id))).first():
                    session.add(Marriage(person_a_id=person_id, person_b_id=spouse_id))
            
            for parent_id in relatives.get('parents', []):
                 if not session.query(Relationship).filter_by(parent_id=parent_id, child_id=person_id).first():
                    session.add(Relationship(parent_id=parent_id, child_id=person_id))
        
        session.commit()
        print("Import z ZIG-JSON zakończony pomyślnie.")
    except Exception as e:
        print(f"Błąd podczas importu ZIG-JSON: {e}", file=sys.stderr)
        session.rollback()


# --- Główna funkcja CLI ---

def main():
    parser = argparse.ArgumentParser(description="zigedcom v3 - Zaawansowane narzędzie genealogiczne.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--db-url', required=True, help="URL do bazy danych (np. 'sqlite:///family_tree_v3.db')")
    
    # Grupy argumentów
    imp = parser.add_argument_group('Import Options')
    imp.add_argument('--import-gedcom', type=str, help="Importuj dane z pliku GEDCOM.")
    imp.add_argument('--import-csv', type=str, help="Importuj dane z pliku CSV.")
    imp.add_argument('--import-zig-json', type=str, help="Importuj dane z pliku ZIG-JSON.")
    
    exp = parser.add_argument_group('Export Options')
    exp.add_argument('--export-gedcom', type=str, help="Eksportuj dane do pliku GEDCOM (częściowo zaimplementowane).")
    exp.add_argument('--export-csv', type=str, help="Eksportuj dane do pliku CSV.")
    exp.add_argument('--export-zig-json', type=str, help="Eksportuj dane do pliku ZIG-JSON.")
    
    args = parser.parse_args()
    Session = get_session(args.db_url)
    session = Session()

    try:
        if args.import_csv: import_csv_to_db(session, args.import_csv)
        if args.import_gedcom: import_gedcom_to_db(session, args.import_gedcom)
        if args.import_zig_json: import_zig_json_to_db(session, args.import_zig_json)
        
        if args.export_csv: print("Eksport do CSV nie został jeszcze w pełni zrefaktoryzowany dla v3.") # export_db_to_csv(session, args.export_csv)
        if args.export_gedcom: export_db_to_gedcom(session, args.export_gedcom)
        if args.export_zig_json: export_db_to_zig_json(session, args.export_zig_json)

    finally:
        session.close()

if __name__ == "__main__":
    try:
        import sqlalchemy, fuzzywuzzy
    except ImportError:
        print("Brak wymaganych bibliotek. Zainstaluj: pip install sqlalchemy psycopg2-binary fuzzywuzzy \"python-levenshtein>=0.12\"", file=sys.stderr)
        sys.exit(1)
    
    main()
