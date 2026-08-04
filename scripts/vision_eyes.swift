import Foundation
import Vision
import AppKit

// DeepSight native eyes v2.1: added `box:` lines for detected-object bounding boxes.
// Apple Vision framework, zero model downloads.
// Usage: vision_eyes <image_path>
// Emits: OCR text (native + 2x upscaled pass for small text), scene
//        classification, attention/objectness saliency, face rectangles +
//        landmarks (roll/yaw/pitch) + capture quality, human rectangles,
//        body pose (joints, arms-up), rectangle counts, animal species,
//        dominant color palette, and `box:` lines for every detected object
//        (faces, humans, animals, text regions, rectangles, salient objects)
//        with normalized coordinates for spatial grounding.
// Proven on macOS 26.4.1 / Xcode SDK 26.5. Compile:
//   env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
//       swiftc -target arm64-apple-macos14 vision_eyes.swift -o vision_eyes

func log(_ s: String) {
    print(s)
    fflush(stdout)
}

/// Emit a `box:` line for spatial grounding: type, confidence, normalized x/y/w/h, label.
/// Coordinates are normalized 0.0-1.0 (Apple Vision convention, origin = bottom-left).
func emitBox(type: String, conf: Float, x: Float, y: Float, w: Float, h: Float, label: String) {
    // Normalized coordinates: origin bottom-left. Python parser flips Y.
    log("box:\(type):\(String(format: "%.4f", conf)):\(String(format: "%.4f", x)):\(String(format: "%.4f", y)):\(String(format: "%.4f", w)):\(String(format: "%.4f", h)):\(label)")
}

guard CommandLine.arguments.count > 1 else { log("usage: vision_eyes <image>"); exit(1) }
let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    log("ERROR: cannot load image"); exit(1)
}
log("loaded \(cg.width)x\(cg.height)")

func run(_ handler: VNImageRequestHandler, _ requests: [VNRequest], label: String) {
    do {
        try handler.perform(requests)
    } catch {
        log("\(label) perform error: \(error)")
    }
}

// Upscale helper: 2x render for small-text OCR (jerseys, captions, labels).
func upscaled(_ cg: CGImage, scale: CGFloat) -> CGImage? {
    let w = Int(CGFloat(cg.width) * scale)
    let h = Int(CGFloat(cg.height) * scale)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()
}

// 1. OCR — native pass + 2x upscale pass, deduped, with bounding boxes
// The normalized bounding box from Apple Vision has origin at bottom-left.
// We emit it as-is; the Python parser converts to top-left (% of image).
var ocrSeen: Set<String> = []
func ocrPass(_ image: CGImage, tag: String) {
    let req = VNRecognizeTextRequest { req, err in
        if let err = err { log("OCR ERROR: \(err)") }
        guard let obs = req.results as? [VNRecognizedTextObservation] else { return }
        for o in obs {
            if let t = o.topCandidates(1).first {
                let s = t.string.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty && !ocrSeen.contains(s) {
                    ocrSeen.insert(s)
                    log("  \(s)")
                    let b = o.boundingBox
                    emitBox(type: "text", conf: 1.0,
                            x: Float(b.minX), y: Float(b.minY),
                            w: Float(b.width), h: Float(b.height),
                            label: s.replacingOccurrences(of: "\n", with: "⏎"))
                }
            }
        }
    }
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.minimumTextHeight = 0.003
    run(VNImageRequestHandler(cgImage: image, options: [:]), [req], label: "OCR-\(tag)")
}
ocrPass(cg, tag: "1x")
if let up = upscaled(cg, scale: 2) { ocrPass(up, tag: "2x") }

// 2. Scene classification (on-device classifier, no downloads)
let clsReq = VNClassifyImageRequest { req, err in
    if let err = err { log("CLASS ERROR: \(err)") }
    if let obs = req.results as? [VNClassificationObservation] {
        let top = obs.prefix(5).map { "\($0.identifier)(\(String(format: "%.2f", $0.confidence)))" }
        log("scene: \(top.joined(separator: ", "))")
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [clsReq], label: "classify")

// 3. Attention saliency
let attReq = VNGenerateAttentionBasedSaliencyImageRequest { req, err in
    if let err = err { log("ATTN ERROR: \(err)") }
    if let obs = req.results?.first as? VNSaliencyImageObservation {
        let map = obs.pixelBuffer
        log("attn map: \(CVPixelBufferGetWidth(map))x\(CVPixelBufferGetHeight(map))")
        if let objs = obs.salientObjects { log("attn salient objects: \(objs.count)") }
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [attReq], label: "attn saliency")

// 4. Objectness
let objReq = VNGenerateObjectnessBasedSaliencyImageRequest { req, err in
    if let err = err { log("OBJ ERROR: \(err)") }
    if let obs = req.results?.first as? VNSaliencyImageObservation {
        let map = obs.pixelBuffer
        log("obj map: \(CVPixelBufferGetWidth(map))x\(CVPixelBufferGetHeight(map))")
        if let objs = obs.salientObjects { log("obj salient objects: \(objs.count)") }
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [objReq], label: "obj saliency")

// 4b. Sports/equipment concepts — classify full frame + salient-region crops,
// map classifier identifiers through a sport taxonomy. On-device, zero downloads.
let SPORT_MAP: [String: String] = [
    "baseball": "baseball", "baseball bat": "baseball", "baseball glove": "baseball",
    "baseball hat": "baseball",
    "basketball": "basketball", "sports ball": "basketball", "basketball hoop": "basketball",
    "american football": "american football", "football helmet": "american football",
    "football": "american football", "helmet": "american football",
    "soccer ball": "soccer", "soccer": "soccer",
    "surfboard": "surfing",
    "volleyball": "volleyball", "tennis ball": "tennis", "tennis racket": "tennis",
    "golf ball": "golf", "golf club": "golf",
    "ice hockey": "ice hockey", "hockey": "ice hockey", "hockey puck": "ice hockey",
    "hockey rink": "ice hockey",
    "hockey stick": "ice hockey", "rink": "ice hockey",
    "ice skates": "ice hockey",
    "sports equipment": "sports equipment",
]
func normId(_ s: String) -> String {
    s.lowercased().replacingOccurrences(of: "_", with: " ").trimmingCharacters(in: .whitespaces)
}
var sportScores: [String: Double] = [:]
func upscale(_ img: CGImage, scale: CGFloat) -> CGImage {
    let w = Int(CGFloat(img.width) * scale)
    let h = Int(CGFloat(img.height) * scale)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: 0, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        return img
    }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? img
}

func classifySports(_ image: CGImage, tag: String) {
    let req = VNClassifyImageRequest { req, err in
        if let err = err { log("SPORTS ERROR (\(tag)): \(err)") }
        guard let obs = req.results as? [VNClassificationObservation] else { return }
        var raw: [String] = []
        for o in obs.prefix(30) {
            let id = normId(o.identifier)
            if o.confidence >= 0.10, let sport = SPORT_MAP[id] {
                sportScores[sport, default: 0] = max(sportScores[sport] ?? 0, Double(o.confidence))
            }
            if o.confidence > 0.05 { raw.append("\(id)(\(String(format: "%.2f", o.confidence)))") }
        }
        if !raw.isEmpty { log("sports raw \(tag): " + raw.prefix(20).joined(separator: ", ")) }
    }
    run(VNImageRequestHandler(cgImage: image, options: [:]), [req], label: "sports-\(tag)")
}
classifySports(cg, tag: "full")
if let sal = objReq.results?.first as? VNSaliencyImageObservation {
    let objs = sal.salientObjects ?? []
    var cropRects: [CGRect] = []
    for o in objs {
        let b = o.boundingBox
        let area = b.width * b.height
        if area > 0.005 {
            cropRects.append(b)
            emitBox(type: "salient", conf: Float(o.confidence),
                    x: Float(b.minX), y: Float(b.minY),
                    w: Float(b.width), h: Float(b.height),
                    label: "salient_object")
        }
    }
    cropRects.sort { $0.width * $0.height > $1.width * $1.height }
    for box in cropRects.prefix(5) {
        let px = CGRect(x: box.minX * CGFloat(cg.width),
                        y: (1 - box.maxY) * CGFloat(cg.height),
                        width: box.width * CGFloat(cg.width),
                        height: box.height * CGFloat(cg.height))
        if let crop = cg.cropping(to: px) {
            log("sports region box: \(Int(px.width))x\(Int(px.height)) @(\(Int(px.minX)),\(Int(px.minY)))")
            classifySports(upscale(crop, scale: 2), tag: "region")
            let cw = px.width / 2, ch = px.height / 2
            for i in 0..<2 {
                for j in 0..<2 {
                    let cell = CGRect(x: px.minX + CGFloat(i) * cw,
                                      y: px.minY + CGFloat(j) * ch,
                                      width: cw, height: ch)
                    if let c2 = cg.cropping(to: cell) {
                        classifySports(upscale(c2, scale: 2), tag: "cell")
                    }
                }
            }
        }
    }
}
if sportScores.isEmpty {
    log("sports: none")
} else {
    let line = sportScores.sorted { $0.value > $1.value }
        .map { "\($0.key)(\(String(format: "%.2f", $0.value)))" }.joined(separator: ", ")
    log("sports: " + line)
}

// 5. Faces — rectangles + landmarks (roll/yaw/pitch) + bounding boxes
let faceReq = VNDetectFaceLandmarksRequest { req, err in
    if let err = err { log("FACE ERROR: \(err)") }
    guard let obs = req.results as? [VNFaceObservation] else { return }
    log("faces: \(obs.count)")
    for (i, f) in obs.enumerated() {
        let roll = f.roll.map { String(format: "%.0f°", $0.doubleValue * 180 / .pi) } ?? "?"
        let yaw = f.yaw.map { String(format: "%.0f°", $0.doubleValue * 180 / .pi) } ?? "?"
        let pitch = f.pitch.map { String(format: "%.0f°", $0.doubleValue * 180 / .pi) } ?? "?"
        log("face attr: \(i): roll=\(roll) yaw=\(yaw) pitch=\(pitch)")
        let b = f.boundingBox
        emitBox(type: "face", conf: Float(f.confidence),
                x: Float(b.minX), y: Float(b.minY),
                w: Float(b.width), h: Float(b.height),
                label: "face")
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [faceReq], label: "faces")

// 5b. Face capture quality (0-1, sharpness/lighting score)
let capReq = VNDetectFaceCaptureQualityRequest { req, err in
    if let err = err { log("CAP ERROR: \(err)") }
    if let obs = req.results as? [VNFaceObservation] {
        let qs = obs.compactMap { $0.faceCaptureQuality.map { String(format: "%.2f", $0) } }
        if !qs.isEmpty { log("face quality: \(qs.joined(separator: ", "))") }
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [capReq], label: "face quality")

// 6. Humans (bounding boxes)
let humanReq = VNDetectHumanRectanglesRequest { req, err in
    if let err = err { log("HUMAN ERROR: \(err)") }
    if let obs = req.results as? [VNHumanObservation] {
        log("humans: \(obs.count)")
        for h in obs {
            let b = h.boundingBox
            emitBox(type: "human", conf: Float(h.confidence),
                    x: Float(b.minX), y: Float(b.minY),
                    w: Float(b.width), h: Float(b.height),
                    label: "human")
        }
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [humanReq], label: "humans")

// 6b. Body pose — joint counts + arms-up (shooting/celebrating) heuristic
let poseReq = VNDetectHumanBodyPoseRequest { req, err in
    if let err = err { log("POSE ERROR: \(err)") }
    guard let obs = req.results as? [VNHumanBodyPoseObservation] else { return }
    log("pose: \(obs.count) human(s)")
    for (i, o) in obs.enumerated() {
        guard let pts = try? o.recognizedPoints(.all) else { continue }
        let joints = pts.filter { $0.value.confidence > 0.3 }.keys
        var armsUp = false
        if let lw = pts[.leftWrist], let rw = pts[.rightWrist],
           let ls = pts[.leftShoulder], let rs = pts[.rightShoulder],
           lw.confidence > 0.3, rw.confidence > 0.3,
           ls.confidence > 0.3, rs.confidence > 0.3 {
            let wristY = max(lw.location.y, rw.location.y)
            let shoulderY = min(ls.location.y, rs.location.y)
            armsUp = wristY > shoulderY + 0.1
        }
        log("pose \(i): joints=\(joints.count) arms_up=\(armsUp)")
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [poseReq], label: "body pose")

// 7. Rectangles (screens, docs, cards) — with bounding boxes
let rectReq = VNDetectRectanglesRequest { req, err in
    if let err = err { log("RECT ERROR: \(err)") }
    if let obs = req.results as? [VNRectangleObservation] {
        log("rectangles: \(obs.count)")
        for r in obs {
            // Compute axis-aligned bounding box from corner points
            let xs = [r.topLeft.x, r.topRight.x, r.bottomLeft.x, r.bottomRight.x]
            let ys = [r.topLeft.y, r.topRight.y, r.bottomLeft.y, r.bottomRight.y]
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            emitBox(type: "rect", conf: Float(r.confidence),
                    x: Float(minX), y: Float(minY),
                    w: Float(maxX - minX), h: Float(maxY - minY),
                    label: "rectangle")
        }
    }
}
rectReq.minimumSize = 0.2
rectReq.minimumAspectRatio = 0.25
rectReq.maximumObservations = 8
run(VNImageRequestHandler(cgImage: cg, options: [:]), [rectReq], label: "rectangles")

// 8. Animals — with bounding boxes
let animalReq = VNRecognizeAnimalsRequest { req, err in
    if let err = err { log("ANIMAL ERROR: \(err)") }
    if let obs = req.results as? [VNRecognizedObjectObservation] {
        let names = obs.compactMap { $0.labels.first?.identifier }
        log("animals: \(names.isEmpty ? "none" : names.joined(separator: ", "))")
        for a in obs {
            if let label = a.labels.first {
                let b = a.boundingBox
                emitBox(type: "animal", conf: Float(label.confidence),
                        x: Float(b.minX), y: Float(b.minY),
                        w: Float(b.width), h: Float(b.height),
                        label: label.identifier)
            }
        }
    }
}
run(VNImageRequestHandler(cgImage: cg, options: [:]), [animalReq], label: "animals")

// 9. Dominant colors — downsample to 32x32, take top exact-color clusters
let w = 32, h = 32
var pixels = [UInt8](repeating: 0, count: w * h * 4)
if let ctx = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                       bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var buckets: [UInt32: Int] = [:]
    var opaque = 0
    for i in stride(from: 0, to: w * h * 4, by: 4) {
        let a = pixels[i + 3]
        if a < 128 { continue }
        opaque += 1
        let r = UInt32(pixels[i]) >> 3
        let g = UInt32(pixels[i + 1]) >> 3
        let b = UInt32(pixels[i + 2]) >> 3
        let key = (r << 10) | (g << 5) | b
        buckets[key, default: 0] += 1
    }
    let total = max(opaque, 1)
    let top = buckets.sorted { $0.value > $1.value }.prefix(6)
    let hexes = top.map { (k, v) -> String in
        let r = ((k >> 10) & 0x1F) << 3
        let g = ((k >> 5) & 0x1F) << 3
        let b = (k & 0x1F) << 3
        let hex = String(format: "#%02X%02X%02X", r, g, b)
        return "\(hex)(\(String(format: "%.0f%%", Double(v) / Double(total) * 100)))"
    }
    log("colors: " + hexes.joined(separator: ", "))
}

log("ALL DONE")
