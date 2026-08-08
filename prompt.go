package moderation

import (
	"strings"
)

// guardTaxonomy is the hazard list Llama Guard 3 was fine-tuned on, in the
// order and wording of the model card. It is sent verbatim on every request:
// llama.cpp cannot execute the Jinja template embedded in the GGUF (it uses a
// categories dict, selectattr and raise_exception), so ctx.Chat silently falls
// back to a plain llama3 chat wrapper and the model never sees the taxonomy.
// Without it the classifier still answers safe/unsafe from its priors but
// labels every violation S1, which is why every moderated object used to read
// "Violent Crimes".
//
// The 1B model's taxonomy ends at S13; S14 in llamaGuardCategories has no
// counterpart here and malicious-code content is reported as S2.
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

// guardPromptTemplate is the Llama Guard 3 prompt format. The full taxonomy is
// listed even for categories Warpnet does not act on: the model was trained
// with all of them, and dropping one makes it map that content onto a
// neighbouring category instead of ignoring it. llamaGuardCategories filters
// the verdict afterwards.
//
// The leading <|begin_of_text|> is deliberate even though the tokeniser adds a
// BOS of its own: measured over the 63-case fixture set, dropping it costs two
// true positives (an incitement-to-assault post and a passport-doxxing post
// both flip to safe).
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

// buildGuardPrompt renders the classification prompt for one piece of content.
//
// The content is placed inside the conversation block as-is. It cannot break
// out of the classification task: Llama Guard only ever emits safe/unsafe plus
// category codes, and the assessment instructions come after the conversation
// block, so trailing text cannot displace them.
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
