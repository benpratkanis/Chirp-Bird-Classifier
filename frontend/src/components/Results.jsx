import React from 'react';

const Results = ({ results, spectrogram }) => {
    if (!results) return null;

    const topResult = results[0];

    return (
        <div className="w-full max-w-2xl mx-auto p-6 bg-white rounded-2xl shadow-lg mt-6 animate-fade-in">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Analysis Results</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex flex-col items-center">
                    <h3 className="text-lg font-semibold text-gray-600 mb-2">Spectrogram</h3>
                    {spectrogram && (
                        <img
                            src={spectrogram}
                            alt="Spectrogram"
                            className="rounded-lg shadow-md border border-gray-200 w-full object-cover"
                        />
                    )}
                </div>

                <div>
                    <h3 className="text-lg font-semibold text-gray-600 mb-2">Predictions</h3>
                    <div className="space-y-3">
                        {results.map((res, idx) => (
                            <div key={idx} className="relative pt-1">
                                <div className="flex mb-2 items-center justify-between">
                                    <div>
                                        <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                                            {res.species}
                                        </span>
                                    </div>
                                    <div className="text-right">
                                        <span className="text-xs font-semibold inline-block text-blue-600">
                                            {(res.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
                                    <div
                                        style={{ width: `${res.probability * 100}%` }}
                                        className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 text-center">
                <p className="text-gray-500">
                    Most likely: <span className="font-bold text-gray-800">{topResult.species}</span>
                </p>
            </div>
        </div>
    );
};

export default Results;
