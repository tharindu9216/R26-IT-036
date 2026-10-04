import type { SupportContact } from "../types";

interface Props {
  contacts?: SupportContact[];
}

/**
 * Contacts for a turn the backend escalated as extremely negative.
 *
 * The reply text already names them — this repeats them as a panel because a
 * distressed reader skims, and a phone number buried in a paragraph is a
 * number nobody dials. The list comes from the backend's configured contacts;
 * the language model never produces one.
 */
export function SupportContacts({ contacts }: Props) {
  if (!contacts?.length) return null;
  return (
    <div className="support-contacts">
      <span className="support-contacts-title">Talk to someone now</span>
      <ul>
        {contacts.map((contact) => (
          <li key={contact.phone}>
            <span className="support-contact-name">{contact.name}</span>
            <a className="support-contact-phone" href={`tel:${contact.phone}`}>
              {contact.phone}
            </a>
            {contact.note && <span className="support-contact-note">{contact.note}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
