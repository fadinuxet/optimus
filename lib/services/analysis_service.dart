import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../config/api_config.dart';

class AnalysisService {
  static String get baseUrl => ApiConfig.baseUrl;

  Future<Map<String, dynamic>> analyzeRefractiveIndex(
      File imageFile, double angle) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/analyze/refractive-index'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
        ),
      );

      request.fields['angle'] = angle.toString();

      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return json.decode(responseData);
      } else {
        throw Exception('Failed to analyze image: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error analyzing image: $e');
    }
  }

  Future<Map<String, dynamic>> analyzeCoating(File imageFile) async {
    final uri = Uri.parse('$baseUrl/analyze/coating');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));
    final response = await request.send();
    final responseBody = await response.stream.bytesToString();
    if (response.statusCode == 200) {
      return json.decode(responseBody);
    } else {
      final error = json.decode(responseBody);
      if (error is Map && error.containsKey('reason')) {
        throw AnalysisQualityException(error['reason']);
      }
      throw Exception('Failed to analyze coating: ${response.reasonPhrase}');
    }
  }

  Future<Map<String, dynamic>> analyzeComposition(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/analyze/composition'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
        ),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return json.decode(responseData);
      } else {
        throw Exception(
            'Failed to analyze composition: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error analyzing composition: $e');
    }
  }

  Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeFingerprint(
    File macroImage, {
    File? video,
    String? lensId,
  }) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/fingerprint'),
    );

    request.files.add(
      await http.MultipartFile.fromPath('macro_image', macroImage.path),
    );

    if (video != null) {
      request.files.add(
        await http.MultipartFile.fromPath('video', video.path),
      );
    }

    if (lensId != null) {
      request.fields['lens_id'] = lensId;
    }

    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    if (response.statusCode == 200) {
      return json.decode(responseData);
    } else {
      final error = json.decode(responseData);
      if (error is Map && error.containsKey('reason')) {
        throw AnalysisQualityException(error['reason']);
      }
      throw Exception(
          'Failed to analyze fingerprint: ${response.reasonPhrase}');
    }
  }

  Future<Map<String, dynamic>> authenticateLens(
    File macroImage, {
    File? video,
    required String lensId,
  }) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/authenticate'),
    );

    request.files.add(
      await http.MultipartFile.fromPath('macro_image', macroImage.path),
    );

    if (video != null) {
      request.files.add(
        await http.MultipartFile.fromPath('video', video.path),
      );
    }

    request.fields['lens_id'] = lensId;

    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    if (response.statusCode == 200) {
      return json.decode(responseData);
    } else {
      final error = json.decode(responseData);
      if (error is Map && error.containsKey('reason')) {
        throw AnalysisQualityException(error['reason']);
      }
      throw Exception('Failed to authenticate lens: ${response.reasonPhrase}');
    }
  }

  Future<List<Map<String, dynamic>>> listLenses() async {
    final response = await http.get(Uri.parse('$baseUrl/api/lenses'));

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return List<Map<String, dynamic>>.from(data['lenses']);
    } else {
      throw Exception('Failed to list lenses: ${response.reasonPhrase}');
    }
  }

  Future<Map<String, dynamic>> authenticateAnyLens(File imageFile) async {
    final uri = Uri.parse('$baseUrl/api/authenticate_any');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(
          await http.MultipartFile.fromPath('macro_image', imageFile.path));
    final response = await request.send();
    final responseBody = await response.stream.bytesToString();
    if (response.statusCode == 200) {
      return jsonDecode(responseBody);
    } else {
      throw Exception('Authentication failed: $responseBody');
    }
  }
}

class AnalysisQualityException implements Exception {
  final String message;
  AnalysisQualityException(this.message);
  @override
  String toString() => message;
}
