import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import '../services/analysis_service.dart';

class AnalysisTestScreen extends StatefulWidget {
  const AnalysisTestScreen({super.key});

  @override
  State<AnalysisTestScreen> createState() => _AnalysisTestScreenState();
}

class _AnalysisTestScreenState extends State<AnalysisTestScreen> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  bool _isCapturing = false;
  bool _isAnalyzing = false;
  bool _isFlashOn = false;
  String? _capturedImagePath;
  final AnalysisService _analysisService = AnalysisService();
  Map<String, dynamic>? _analysisResults;
  List<Map<String, dynamic>> _registeredLenses = [];
  bool _isRegistering = false;
  final _lensIdController = TextEditingController();
  bool _isAuthenticatingAny = false;
  bool? _lastAuthResult;
  bool _hasCaptured = false;
  bool isTeamMode = true; // Set to false for public release
  bool _showAnalysisPanel = true;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadRegisteredLenses();
  }

  Future<void> _loadRegisteredLenses() async {
    try {
      final lenses = await _analysisService.listLenses();
      setState(() {
        _registeredLenses = lenses;
      });
    } catch (e) {
      debugPrint('Error loading lenses: $e');
    }
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      return;
    }

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _toggleFlash() async {
    if (_controller == null) return;
    try {
      _isFlashOn = !_isFlashOn;
      await _controller!
          .setFlashMode(_isFlashOn ? FlashMode.torch : FlashMode.off);
      setState(() {});
    } catch (e) {
      debugPrint('Error toggling flash: $e');
    }
  }

  Future<void> _captureImage() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }
    setState(() {
      _isCapturing = true;
    });
    try {
      final XFile image = await _controller!.takePicture();
      setState(() {
        _capturedImagePath = image.path;
        _isCapturing = false;
        _hasCaptured = true;
        _lastAuthResult = null;
      });
    } catch (e) {
      setState(() {
        _isCapturing = false;
        _hasCaptured = false;
      });
      _showError('Failed to capture image: $e');
    }
  }

  void _resetCapture() {
    setState(() {
      _capturedImagePath = null;
      _hasCaptured = false;
      _lastAuthResult = null;
    });
  }

  Future<void> _registerLens() async {
    if (_capturedImagePath == null) {
      _showError('Please capture an image first');
      return;
    }

    final lensId = _lensIdController.text.trim();
    if (lensId.isEmpty) {
      _showError('Please enter a lens ID');
      return;
    }

    setState(() {
      _isRegistering = true;
    });

    try {
      // Analyze the image if not already analyzed
      if (_analysisResults == null) {
        _analysisResults = await _analysisService.analyzeFingerprint(
          File(_capturedImagePath!),
        );
      }
      await _analysisService.analyzeFingerprint(
        File(_capturedImagePath!),
        lensId: lensId,
      );

      await _loadRegisteredLenses();

      setState(() {
        _isRegistering = false;
      });

      _showSuccess('Lens registered successfully');
    } catch (e) {
      setState(() {
        _isRegistering = false;
      });
      _showError('Failed to register lens: $e');
    }
  }

  Future<void> _authenticateLens(String lensId) async {
    if (_capturedImagePath == null) {
      _showError('Please capture an image first');
      return;
    }

    setState(() {
      _isAnalyzing = true;
    });

    try {
      final results = await _analysisService.authenticateLens(
        File(_capturedImagePath!),
        lensId: lensId,
      );

      setState(() {
        _isAnalyzing = false;
      });

      _showAuthenticationResults(results);
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
      });
      _showError('Failed to authenticate lens: $e');
    }
  }

  Future<void> _authenticateAnyLens() async {
    if (_capturedImagePath == null) {
      _showError('Please capture an image first');
      return;
    }
    setState(() {
      _isAuthenticatingAny = true;
      _lastAuthResult = null;
    });
    try {
      final results = await _analysisService.authenticateAnyLens(
        File(_capturedImagePath!),
      );
      setState(() {
        _isAuthenticatingAny = false;
        _lastAuthResult = results['authenticated'] == true;
      });
      _showAnyAuthenticationResults(results);
    } catch (e) {
      setState(() {
        _isAuthenticatingAny = false;
        _lastAuthResult = null;
      });
      _showError('Failed to authenticate lens: $e');
    }
  }

  void _showAuthenticationResults(Map<String, dynamic> results) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(results['authenticated'] ? 'Authentic Lens' : 'Fake Lens'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Similarity Score: ${(results['similarity_score'] * 100).toStringAsFixed(1)}%'),
            Text('Quality: ${results['quality']}'),
            Text('Score: ${results['score']}'),
            Text('Description: ${results['description']}'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showAnyAuthenticationResults(Map<String, dynamic> results) {
    final isAuth = results['authenticated'] == true;
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: isAuth ? Colors.green[50] : Colors.red[50],
        title: Text(isAuth ? 'Authentic Lens' : 'Not Authentic',
            style: TextStyle(color: isAuth ? Colors.green : Colors.red)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
                'Similarity Score: ${(results['similarity_score'] * 100).toStringAsFixed(1)}%'),
            if (results['matched_lens_id'] != null)
              Text('Matched Lens ID: ${results['matched_lens_id']}'),
            Text('Quality: ${results['quality']}'),
            Text('Score: ${results['score']}'),
            Text('Description: ${results['description']}'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showAnalysisResults() {
    if (_analysisResults == null || _analysisResults!.isEmpty) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Analysis Results'),
          content: const Text('No analysis results available.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );
      return;
    }

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Analysis Results'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (_analysisResults!['fingerprint_hash'] != null)
                _buildResultItem(
                  'Fingerprint Hash',
                  _analysisResults!['fingerprint_hash'].toString(),
                ),
              if (_analysisResults!['quality'] != null)
                _buildResultItem(
                  'Quality',
                  _analysisResults!['quality'].toString(),
                ),
              if (_analysisResults!['score'] != null)
                _buildResultItem(
                  'Score',
                  _analysisResults!['score'].toString(),
                ),
              if (_analysisResults!['fingerprint_vector'] != null)
                _buildResultItem(
                  'Feature Vector',
                  (_analysisResults!['fingerprint_vector'] is List)
                      ? '${(_analysisResults!['fingerprint_vector'] as List).length} dimensions'
                      : 'N/A',
                ),
              if (_analysisResults!['description'] != null)
                _buildResultItem(
                  'Description',
                  _analysisResults!['description'].toString(),
                ),
              const SizedBox(height: 16),
              TextField(
                controller: _lensIdController,
                decoration: const InputDecoration(
                  labelText: 'Lens ID',
                  hintText: 'Enter ID to register this lens',
                ),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _isRegistering ? null : _registerLens,
                child: _isRegistering
                    ? const CircularProgressIndicator()
                    : const Text('Register Lens'),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Widget _buildResultItem(String title, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showWarning(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Capture Quality Too Low'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showSuccess(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
      ),
    );
  }

  void _toggleMode() {
    setState(() {
      isTeamMode = !isTeamMode;
      _resetCapture();
    });
  }

  void _toggleAnalysisPanel() {
    setState(() {
      _showAnalysisPanel = !_showAnalysisPanel;
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _lensIdController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
            isTeamMode ? 'Lens Auth (Team Mode)' : 'Lens Auth (User Mode)'),
        actions: [
          IconButton(
            icon: Icon(_isFlashOn ? Icons.flash_on : Icons.flash_off),
            onPressed: _toggleFlash,
          ),
          IconButton(
            icon: Icon(isTeamMode ? Icons.group : Icons.person),
            tooltip: isTeamMode ? 'Switch to User Mode' : 'Switch to Team Mode',
            onPressed: _toggleMode,
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(
          children: [
            // Camera preview or captured image
            Positioned.fill(
              child: _hasCaptured && _capturedImagePath != null
                  ? Image.file(
                      File(_capturedImagePath!),
                      fit: BoxFit.cover,
                    )
                  : CameraPreview(_controller!),
            ),
            // Guide overlay
            if (!_hasCaptured)
              Positioned.fill(
                child: CustomPaint(
                  painter: AnalysisGuidePainter(),
                ),
              ),
            // Green/Red overlay after authentication
            if (_lastAuthResult != null)
              Positioned.fill(
                child: Container(
                  color: _lastAuthResult == true
                      ? Colors.green.withOpacity(0.2)
                      : Colors.red.withOpacity(0.2),
                ),
              ),
            // Loading overlay
            if (_isAnalyzing)
              Positioned.fill(
                child: Container(
                  color: Colors.black54,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      Container(
                        width: double.infinity,
                        color: Colors.blue.withOpacity(0.8),
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        child: Column(
                          children: [
                            const LinearProgressIndicator(
                              backgroundColor: Colors.white24,
                              valueColor:
                                  AlwaysStoppedAnimation<Color>(Colors.white),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Analyzing Lens...',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const Spacer(),
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      const Text(
                        'Processing...',
                        style: TextStyle(color: Colors.white),
                      ),
                    ],
                  ),
                ),
              ),
            // Bottom controls
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Padding(
                padding: const EdgeInsets.only(
                    bottom: 32, left: 24, right: 24, top: 8),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (_hasCaptured)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Expanded(
                            child: ElevatedButton(
                              onPressed: _resetCapture,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.grey[300],
                                foregroundColor: Colors.black,
                                padding:
                                    const EdgeInsets.symmetric(vertical: 16),
                              ),
                              child: const Text('Recapture'),
                            ),
                          ),
                          if (!isTeamMode) ...[
                            const SizedBox(width: 16),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: (_isAuthenticatingAny ||
                                        _capturedImagePath == null)
                                    ? null
                                    : _authenticateAnyLens,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: _lastAuthResult == null
                                      ? Theme.of(context).primaryColor
                                      : (_lastAuthResult == true
                                          ? Colors.green
                                          : Colors.red),
                                  foregroundColor: Colors.white,
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 16),
                                ),
                                child: const Text('Check Authenticity'),
                              ),
                            ),
                          ],
                        ],
                      ),
                    if (isTeamMode && _hasCaptured)
                      Padding(
                        padding: const EdgeInsets.only(top: 24.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            // Show/hide analysis results
                            if (_analysisResults != null && _showAnalysisPanel)
                              Card(
                                color: Colors.white,
                                elevation: 4,
                                margin: const EdgeInsets.only(bottom: 16),
                                child: Padding(
                                  padding: const EdgeInsets.all(16.0),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      if (_analysisResults![
                                              'fingerprint_hash'] !=
                                          null)
                                        _buildResultItem(
                                          'Fingerprint Hash',
                                          _analysisResults!['fingerprint_hash']
                                              .toString(),
                                        ),
                                      if (_analysisResults!['quality'] != null)
                                        _buildResultItem(
                                          'Quality',
                                          _analysisResults!['quality']
                                              .toString(),
                                        ),
                                      if (_analysisResults!['score'] != null)
                                        _buildResultItem(
                                          'Score',
                                          _analysisResults!['score'].toString(),
                                        ),
                                      if (_analysisResults![
                                              'fingerprint_vector'] !=
                                          null)
                                        _buildResultItem(
                                          'Feature Vector',
                                          (_analysisResults![
                                                  'fingerprint_vector'] is List)
                                              ? '${(_analysisResults!['fingerprint_vector'] as List).length} dimensions'
                                              : 'N/A',
                                        ),
                                      if (_analysisResults!['description'] !=
                                          null)
                                        _buildResultItem(
                                          'Description',
                                          _analysisResults!['description']
                                              .toString(),
                                        ),
                                    ],
                                  ),
                                ),
                              ),
                            if (_analysisResults != null)
                              TextButton(
                                onPressed: _toggleAnalysisPanel,
                                child: Text(_showAnalysisPanel
                                    ? 'Hide Results'
                                    : 'Show Results'),
                              ),
                            Container(
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                    color: Colors.black54, width: 1.2),
                              ),
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 8),
                              margin: const EdgeInsets.only(bottom: 12),
                              child: TextField(
                                controller: _lensIdController,
                                style: const TextStyle(
                                    color: Colors.black,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 18),
                                decoration: const InputDecoration(
                                  labelText: 'Lens ID',
                                  labelStyle: TextStyle(
                                      color: Colors.black87,
                                      fontWeight: FontWeight.bold),
                                  hintText: 'Enter ID to register this lens',
                                  border: InputBorder.none,
                                  filled: false,
                                ),
                              ),
                            ),
                            const SizedBox(height: 8),
                            ElevatedButton(
                              onPressed: _isRegistering ? null : _registerLens,
                              child: _isRegistering
                                  ? const CircularProgressIndicator()
                                  : const Text('Register Lens'),
                            ),
                          ],
                        ),
                      ),
                    if (isTeamMode && _registeredLenses.isNotEmpty)
                      Container(
                        height: 100,
                        margin: const EdgeInsets.only(top: 16),
                        child: ListView.builder(
                          scrollDirection: Axis.horizontal,
                          itemCount: _registeredLenses.length,
                          itemBuilder: (context, index) {
                            final lens = _registeredLenses[index];
                            return Card(
                              margin: const EdgeInsets.only(right: 8),
                              child: InkWell(
                                onTap: () => _authenticateLens(lens['lens_id']),
                                child: Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Text(
                                        'Lens ${lens['lens_id']}',
                                        style: const TextStyle(
                                            fontWeight: FontWeight.bold),
                                      ),
                                      Text('Hash: ${lens['fingerprint_hash']}'),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    if (!_hasCaptured)
                      Center(
                        child: GestureDetector(
                          onTap: _isCapturing ? null : _captureImage,
                          child: Container(
                            width: 80,
                            height: 80,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: Colors.black,
                                width: 4,
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black26,
                                  blurRadius: 8,
                                  offset: Offset(0, 4),
                                ),
                              ],
                            ),
                            child: _isCapturing
                                ? const Center(
                                    child: CircularProgressIndicator())
                                : const Icon(Icons.camera_alt,
                                    size: 40, color: Colors.black),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class AnalysisGuidePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Draw center circle for lens placement
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.3;

    canvas.drawCircle(center, radius, paint);

    // Draw crosshair
    canvas.drawLine(
      Offset(center.dx - radius, center.dy),
      Offset(center.dx + radius, center.dy),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx, center.dy - radius),
      Offset(center.dx, center.dy + radius),
      paint,
    );
  }

  @override
  bool shouldRepaint(AnalysisGuidePainter oldDelegate) => false;
}
