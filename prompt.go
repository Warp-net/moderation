package moderation

import (
	"errors"
	"strings"
)

const guardTaxonomy = `S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.`

const guardPromptTemplate = `<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
` + guardTaxonomy + `
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: %s

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

`

func buildGuardPrompt(content string) string {
	var b strings.Builder
	head, tail, _ := strings.Cut(guardPromptTemplate, "%s")
	b.Grow(len(head) + len(content) + len(tail))
	b.WriteString(head)
	b.WriteString(content)
	b.WriteString(tail)
	return b.String()
}

// llamaGuardCategories maps Llama Guard 3 hazard codes (S1–S14) to
// human-readable reasons. The codes correspond to the taxonomy embedded in the
// model's chat template; the model replies with "unsafe" on the first line
// followed by a comma-separated list of violated codes on the second.
var llamaGuardCategories = map[string]string{
	"S1": "Violent Crimes",
	"S2": "Non-Violent Crimes",
	"S3": "Sex Crimes",
	"S4": "Child Exploitation",
	//"S5":  "Defamation",
	//"S6":  "Specialized Advice",
	"S7": "Privacy",
	//"S8":  "Intellectual Property",
	"S9":  "Indiscriminate Weapons",
	"S10": "Hate",
	"S11": "Self-Harm",
	//"S12": "Sexual Content",
	//"S13": "Elections",
	"S14": "Code Interpreter Abuse",
}

// parseViolationReason turns a Llama Guard "unsafe" response into a
// human-readable, comma-separated reason. The raw model output looks like:
//
//	unsafe
//	S9,S2
//
// llamaGuardCategories acts as an allow-list: codes that are not present (e.g.
// the ones commented out there) are treated as non-violations and skipped, so
// they do not appear in the reason. If every reported code is skipped the
// result is empty and the caller treats the content as safe.
func parseViolationReason(output string) string {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) < 2 {
		return ""
	}
	var reasons []string
	for _, code := range strings.Split(lines[1], ",") {
		code = strings.ToUpper(strings.TrimSpace(code))
		if code == "" {
			continue
		}
		if name, ok := llamaGuardCategories[code]; ok {
			reasons = append(reasons, name)
		}
	}
	return strings.Join(reasons, ", ")
}

func classifyOutput(raw string) (bool, string, error) {
	out := strings.ToLower(strings.TrimSpace(raw))

	switch {
	case strings.HasPrefix(out, "safe"):
		return true, "", nil
	case strings.HasPrefix(out, "unsafe"):
		reason := parseViolationReason(raw)
		if reason == "" {
			return true, "", nil
		}
		return false, reason, nil
	default:
		return true, "", errors.New("unrecognized LLM output: " + out)
	}
}
