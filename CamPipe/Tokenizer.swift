import Foundation

/// CLIP's byte-pair-encoding tokenizer (a port of OpenAI's SimpleTokenizer), which
/// turns typed text into the 77 token ids the YOLOE text encoder takes.
/// `clip_merges.txt` (from tools/export_yoloe.py) holds the merge list; the
/// vocabulary is rebuilt from it the same way CLIP does.
final class CLIPTokenizer {
    static let contextLength = 77
    static let startToken: Int32 = 49406
    static let endToken: Int32 = 49407

    private let byteEncoder: [UInt8: String]
    private let encoder: [String: Int32]
    private let ranks: [String: Int]            // "first second" -> merge rank
    private var cache: [String: [String]] = [:]
    private let pattern = try! NSRegularExpression(
        pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
        options: [.caseInsensitive])

    init(mergesText: String) {
        // bytes_to_unicode(): printable bytes map to themselves, the rest to 256+n.
        var bs = Array(UInt8(ascii: "!")...UInt8(ascii: "~")) + Array(UInt8(0xA1)...UInt8(0xAC)) + Array(UInt8(0xAE)...UInt8(0xFF))
        var cs = bs.map { UInt32($0) }
        var n: UInt32 = 0
        for b in 0...255 where !bs.contains(UInt8(b)) {
            bs.append(UInt8(b))
            cs.append(256 + n)
            n += 1
        }
        let chars = cs.map { String(Character(Unicode.Scalar($0)!)) }
        var be: [UInt8: String] = [:]
        for (b, c) in zip(bs, chars) { be[b] = c }
        byteEncoder = be

        let merges = mergesText.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        var vocab = chars + chars.map { $0 + "</w>" }
        var r: [String: Int] = [:]
        for (i, m) in merges.enumerated() {
            r[m] = i
            vocab.append(m.replacingOccurrences(of: " ", with: ""))
        }
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        var enc: [String: Int32] = [:]
        for (i, v) in vocab.enumerated() { enc[v] = Int32(i) }
        encoder = enc
        ranks = r
    }

    convenience init?(bundle: Bundle = .main) {
        guard let url = bundle.url(forResource: "clip_merges", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        self.init(mergesText: text)
    }

    /// [start] + BPE ids + [end], zero-padded to 77 (cut to fit, keeping [end]).
    func tokenize(_ text: String) -> [Int32] {
        var ids = [Self.startToken] + encode(text) + [Self.endToken]
        if ids.count > Self.contextLength {
            ids = Array(ids.prefix(Self.contextLength))
            ids[Self.contextLength - 1] = Self.endToken
        }
        return ids + Array(repeating: 0, count: Self.contextLength - ids.count)
    }

    func encode(_ text: String) -> [Int32] {
        let cleaned = Self.htmlUnescape(Self.htmlUnescape(text))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
            .lowercased()
        var out: [Int32] = []
        let ns = cleaned as NSString
        for m in pattern.matches(in: cleaned, range: NSRange(location: 0, length: ns.length)) {
            let token = ns.substring(with: m.range)
            let mapped = token.utf8.map { byteEncoder[$0]! }.joined()
            for piece in bpe(mapped) {
                if let id = encoder[piece] { out.append(id) }
            }
        }
        return out
    }

    private func bpe(_ token: String) -> [String] {
        if let hit = cache[token] { return hit }
        var word = token.unicodeScalars.map { String($0) }   // one symbol per mapped byte
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            // The adjacent pair with the lowest merge rank.
            var best: (rank: Int, first: String, second: String)?
            for i in 0..<(word.count - 1) {
                if let r = ranks[word[i] + " " + word[i + 1]], r < (best?.rank ?? Int.max) {
                    best = (r, word[i], word[i + 1])
                }
            }
            guard let pair = best else { break }
            let first = pair.first, second = pair.second
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        cache[token] = word
        return word
    }

    /// The HTML escapes CLIP's cleaning undoes (it runs html.unescape twice).
    private static func htmlUnescape(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var r = s
        for (k, v) in [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"), ("&amp;", "&")] {
            r = r.replacingOccurrences(of: k, with: v)
        }
        return r
    }
}
