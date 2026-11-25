import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct OpenAIDecisionsClient: AgentDecisionClient {
    public typealias Transport = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)

    public var name: String { "OpenAI Decisions" }

    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1")!
    public static let defaultModel = "gpt-6-luna"
    public static let defaultInputUSDPerMillionTokens = 0.10

    private static let abstain = "__axe_no_match__"

    private let apiKey: String
    private let endpoint: URL
    private let transport: Transport

    public init(apiKey: String, baseURL: URL = defaultBaseURL, transport: Transport? = nil) throws {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw OpenAIDecisionsError.missingAPIKey }
        self.apiKey = key
        let versionedBase = baseURL.lastPathComponent == "v1" ? baseURL : baseURL.appendingPathComponent("v1")
        self.endpoint = versionedBase.appendingPathComponent("decisions")
        self.transport = transport ?? { request in
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let response = response as? HTTPURLResponse else {
                throw OpenAIDecisionsError.invalidResponse
            }
            return (data, response)
        }
    }

    public func warmUp() async {
        var request = URLRequest(url: endpoint.deletingLastPathComponent(), timeoutInterval: 5)
        request.httpMethod = "HEAD"
        _ = try? await transport(request)
    }

    public func decide(_ payload: TypeSafeSystemOneRequest) async throws -> TypeSafeSystemOneResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let questions = try payload.questions.sorted(by: { $0.key < $1.key }).map { name, question in
            var choices = try question.criteria.sorted(by: { $0.key < $1.key }).map { value, description in
                Choice(value: value, description: try Self.text(description, encoder: encoder))
            }
            if choices.count == 1 {
                // Decisions requires two options. The only offered literal still needs validation.
                choices.append(Choice(value: Self.abstain, description: "The offered choice does not fit the goal or current state."))
            }
            return Question(name: name, instructions: try Self.text(question.instructions, encoder: encoder), choices: choices)
        }
        let body = Request(
            model: payload.model,
            input: try Self.text(payload.state, encoder: encoder),
            questions: questions
        )

        var request = URLRequest(url: endpoint, timeoutInterval: 25)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        let (data, http) = try await transport(request)
        guard (200..<300).contains(http.statusCode) else {
            throw OpenAIDecisionsError.httpStatus(http.statusCode)
        }

        guard let response = try? JSONDecoder().decode(Response.self, from: data),
              response.answers.count == questions.count else {
            throw OpenAIDecisionsError.invalidResponse
        }
        var answers: [String: TypeSafeChoiceAnswer] = [:]
        for (question, answer) in zip(questions, response.answers) {
            guard answer.name == question.name else { throw OpenAIDecisionsError.invalidResponse }
            guard answer.type != "refusal" else { continue }
            guard answer.type == "choice",
                  let selected = answer.choice,
                  let confidence = answer.confidence,
                  let probabilities = answer.probabilities else {
                throw OpenAIDecisionsError.invalidResponse
            }
            // Detect duplicates before changing the wire array into the policy's dictionary.
            let scores = Dictionary(probabilities.map { ($0.value, $0.probability) }, uniquingKeysWith: { first, _ in first })
            guard scores.count == probabilities.count else { continue }
            var choice = TypeSafeChoiceAnswer(type: answer.type, choice: selected, confidence: confidence, probabilities: scores)
            if question.choices.last?.value == Self.abstain {
                guard selected != Self.abstain,
                      (try? AgentPolicy.validateChoice(
                          choice, named: question.name, allowedChoices: Set(question.choices.map(\.value))
                      )) != nil else {
                    continue
                }
                choice = TypeSafeChoiceAnswer(type: "choice", choice: selected, confidence: confidence, probabilities: [selected: 1])
            }
            answers[question.name] = choice
        }
        return TypeSafeSystemOneResponse(model: response.model, answers: answers, usage: response.usage)
    }

    private static func text(_ value: AgentJSONValue?, encoder: JSONEncoder) throws -> String {
        guard let value else { return "" }
        if let text = value.stringValue { return text }
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    private struct Choice: Encodable {
        let value: String
        let description: String
    }

    private struct Question: Encodable {
        let type = "choice"
        let name: String
        let instructions: String
        let choices: [Choice]
    }

    private struct Request: Encodable {
        let model: String
        let input: String
        let questions: [Question]
    }

    private struct Response: Decodable {
        let model: String
        let answers: [Answer]
        let usage: TypeSafeUsage?
    }

    private struct Answer: Decodable {
        struct Probability: Decodable {
            let value: String
            let probability: Double
        }

        let name: String?
        let type: String
        let choice: String?
        let confidence: Double?
        let probabilities: [Probability]?
    }
}

enum OpenAIDecisionsError: LocalizedError {
    case missingAPIKey
    case invalidResponse
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey: "OpenAI Decisions requires OPENAI_API_KEY."
        case .invalidResponse: "OpenAI Decisions returned an invalid response."
        case .httpStatus(let status): "OpenAI Decisions returned HTTP \(status)."
        }
    }
}
