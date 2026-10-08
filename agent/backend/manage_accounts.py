"""Provision trusted doctor accounts: python manage_accounts.py doctor --email ... --name ..."""
import argparse
import getpass

from auth import accounts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['doctor'])
    parser.add_argument('--email', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    password = getpass.getpass('Password (10–128 characters): ')
    if password != getpass.getpass('Confirm password: '):
        parser.error('Passwords do not match.')
    try:
        account = accounts.create(args.email, args.name, password, args.role)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Created doctor account for {account['email']}.")
