# Data Handling and Privacy Policy

**Document ID:** POL-DATA-002
**Applies to:** All systems that process employee or customer personal data,
including the InsightBot AI assistant and its conversation logs

## 1. Categories of Sensitive Data

This policy treats the following as "Sensitive Personal Data" requiring the
highest level of protection:
- Government identifiers (Social Security Numbers, national ID numbers,
  passport numbers)
- Financial account numbers, credit card numbers, IBANs/routing numbers
- Full names combined with contact details (email address, phone number,
  home address)

## 2. Minimum Necessary Principle

### 2.1 Assistant Responses
The assistant must not include Sensitive Personal Data (as defined in
Section 1) in its responses unless the user is the verified data subject
and the information is strictly necessary to complete the requested task.

### 2.2 Redaction
When Sensitive Personal Data appears in a user's message and is not needed
to answer the request, the assistant should avoid repeating it back verbatim
and should note that the information has been received but will not be
stored or echoed.

## 3. Logging and Retention

### 3.1 Log Scrubbing
Conversation logs retained for quality assurance or audit purposes must be
screened for Sensitive Personal Data. Any log entry containing Sensitive
Personal Data (Section 1) is classified as **high-risk** and must be
reviewed by the Data Protection Officer (DPO).

### 3.2 Credit Card and Bank Data
Under no circumstances should full credit card numbers, bank account
numbers, or IBANs be stored in plaintext logs. Entries containing these are
classified as **critical** and trigger an incident notification per Section 4.

## 4. Incident Notification

Any log entry classified as critical under Section 3.2, or any confirmed
unauthorized disclosure of Sensitive Personal Data, must generate a DPO
incident notification within one business day, per the Continuous
Monitoring Recommendations produced by the audit pipeline.
