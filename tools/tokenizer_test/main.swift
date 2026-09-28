// Checks CamPipe/Tokenizer.swift against CLIP's Python tokenizer.
// usage: tokenizer_test clip_merges.txt tokenizer_tests.json   (run by the Build IPA workflow)
import Foundation

struct Case: Decodable { let text: String; let ids: [Int32] }

let args = CommandLine.arguments
let merges = try String(contentsOfFile: args[1], encoding: .utf8)
let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
let tok = CLIPTokenizer(mergesText: merges)

var failed = 0
for c in cases {
    let got = tok.tokenize(c.text).filter { $0 != 0 }
    if got == c.ids {
        print("ok    \"\(c.text)\" -> \(got)")
    } else {
        print("FAIL  \"\(c.text)\"\n      want \(c.ids)\n      got  \(got)")
        failed += 1
    }
}
print("\(cases.count - failed)/\(cases.count) tokenizer cases match CLIP")
exit(failed == 0 ? 0 : 1)
