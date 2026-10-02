"""Admin commands for accounts (change proposal D8).

    python -m app.auth.cli reset-password <email>
    python -m app.auth.cli list-users

Run from the docuresearch/ directory. Uses the same database as the app
(config.toml / DOCURESEARCH_DB_PATH).
"""

from __future__ import annotations

import argparse
import getpass
import sys

from app.auth.service import AuthError, set_password
from app.config import Settings
from app.store.schema import create_schema, get_connection, set_db_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocuResearch account administration")
    sub = parser.add_subparsers(dest="command", required=True)
    reset = sub.add_parser("reset-password", help="Set a new password for a user")
    reset.add_argument("email")
    sub.add_parser("list-users", help="List registered accounts")
    args = parser.parse_args(argv)

    set_db_path(str(Settings.load().db_path))
    conn = get_connection()
    try:
        create_schema(conn)
        if args.command == "list-users":
            for row in conn.execute("SELECT email, display_name, created_at FROM users "
                                    "ORDER BY created_at"):
                print(f"{row['email']}\t{row['display_name']}\t{row['created_at']}")
            return 0

        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Repeat new password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 1
        try:
            changed = set_password(conn, args.email, password)
        except AuthError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not changed:
            print(f"No account with email {args.email}.", file=sys.stderr)
            return 1
        print(f"Password updated for {args.email}. Existing sessions were signed out.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
