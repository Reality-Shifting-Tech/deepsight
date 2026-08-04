// eval/make_synthetic.swift — deterministic synthetic test images.
// Zero network, zero tokens: the whole point is known ground truth.
//
// Usage: make_synthetic <out.png> <solid|rect|textcard> <color1> <color2> [text lines...]
//   solid:    canvas filled with color1
//   rect:     color2 rectangle centered on color1 background
//   textcard: color2 bold text centered on color1 background (up to 2 lines)

import AppKit

extension NSColor {
    convenience init(hex: String) {
        var s = hex.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("#") { s.removeFirst() }
        var v: UInt64 = 0
        Scanner(string: s).scanHexInt64(&v)
        let r = CGFloat((v >> 16) & 0xFF) / 255.0
        let g = CGFloat((v >> 8) & 0xFF) / 255.0
        let b = CGFloat(v & 0xFF) / 255.0
        self.init(srgbRed: r, green: g, blue: b, alpha: 1)
    }
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    FileHandle.standardError.write("usage: make_synthetic <out.png> <solid|rect|textcard> <color1> <color2> [text...]\n".data(using: .utf8)!)
    exit(2)
}

let outPath = args[1]
let kind = args[2]
let color1 = NSColor(hex: args[3])
let color2 = NSColor(hex: args[4])
let W = 640.0
let H = 480.0

let image = NSImage(size: NSSize(width: W, height: H))
image.lockFocus()

color1.setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()

switch kind {
case "rect":
    color2.setFill()
    NSRect(x: W / 2 - 130, y: H / 2 - 70, width: 260, height: 140).fill()
case "textcard":
    let font = NSFont.boldSystemFont(ofSize: 72)
    let para = NSMutableParagraphStyle()
    para.alignment = .center
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color2,
        .paragraphStyle: para,
    ]
    let lines = Array(args.dropFirst(5))
    var y = H / 2 + (lines.count > 1 ? 45 : 0)
    for line in lines {
        let attr = NSAttributedString(string: line, attributes: attrs)
        let size = attr.size()
        attr.draw(in: NSRect(x: 0, y: y - size.height / 2, width: W, height: size.height))
        y -= 120
    }
default:
    break
}

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let png = rep.representation(using: .png, properties: [:]) else {
    FileHandle.standardError.write("render failed\n".data(using: .utf8)!)
    exit(1)
}
try! png.write(to: URL(fileURLWithPath: outPath))
